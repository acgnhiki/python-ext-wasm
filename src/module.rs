//! The `wasmer.Module` Python object to build WebAssembly modules.

use crate::{
    import::ImportObject,
    instance::{
        exports::{ExportImportKind, ExportedFunctions},
        globals::ExportedGlobals,
        Instance,
    },
    memory::Memory,
    wasi,
    wasmer::{
        core::{
            cache::Artifact,
            module::ExportIndex,
            types::{ExternDescriptor, Type},
        },
        runtime::{self as runtime, validate, Export},
    },
};
use pyo3::{
    exceptions::RuntimeError,
    prelude::*,
    types::{PyAny, PyBytes, PyDict, PyList},
    PyTryFrom,
};
use std::sync::Arc;
use wasmer_wasi;

#[pyclass(unsendable)]
#[text_signature = "(bytes)"]
/// `Module` is a Python class that represents a WebAssembly module.
pub struct Module {
    /// The underlying Rust WebAssembly module.
    pub(crate) inner: Arc<runtime::Module>,
}

#[pymethods]
/// Implement methods on the `Module` Python class.
impl Module {
    /// Check that given bytes represent a valid WebAssembly module.
    ///
    /// # Examples
    ///
    /// ```py
    /// is_valid = Module.validate(wasm_bytes)
    /// ```
    #[text_signature = "(bytes)"]
    #[staticmethod]
    fn validate(bytes: &PyAny) -> PyResult<bool> {
        match <PyBytes as PyTryFrom>::try_from(bytes) {
            Ok(bytes) => Ok(validate(bytes.as_bytes())),
            _ => Ok(false),
        }
    }

    /// Compile bytes into a WebAssembly module.
    ///
    /// # Examples
    ///
    /// ```py
    /// module = Module(wasm_bytes)
    /// ```
    #[new]
    #[allow(clippy::new_ret_no_self)]
    fn new(bytes: &PyAny) -> PyResult<Self> {
        // Read the bytes.
        let bytes = <PyBytes as PyTryFrom>::try_from(bytes)?.as_bytes();

        // Compile the module.
        let module = runtime::compile(bytes).map_err(|error| {
            RuntimeError::py_err(format!("Failed to compile the module:\n    {}", error))
        })?;

        Ok(Self {
            inner: Arc::new(module),
        })
    }

    /// Instantiate the module into an `Instance` Python object. The
    /// `import_object` argument is optional, and can be either a
    /// Python dictionary, or an `ImportObject` (generated by
    /// `module.generate_import_object()`).
    ///
    /// # Examples
    ///
    /// ```py
    /// module = Module(wasm_bytes)
    /// instance = module.instantiate()
    /// ```
    ///
    /// or with an `ImportObject`:
    ///
    /// ```py
    /// import_object = module.generate_import_object()
    /// # do something with `import_object`
    /// instance = module.instantiate(import_object)
    /// ```
    ///
    /// or with a dictionary:
    ///
    /// ```py
    /// instance = module.instantiate({"env": { "sum": sum }})
    /// ```
    #[text_signature = "($self, import_object={})"]
    #[args(import_object = "PyDict::new(_py).as_ref()")]
    fn instantiate(&self, py: Python, import_object: &PyAny) -> PyResult<Py<Instance>> {
        // Instantiate the WebAssembly module, …
        let instance =
            // … with an import object
            if let Ok(import_object) = import_object.downcast::<PyCell<ImportObject>>() {
                let import_object = import_object.borrow();

                self.inner.instantiate(&(*import_object).inner)
            }
            // … with a Python dictionary
            else if let Ok(imported_functions) = import_object.downcast::<PyDict>() {
                let mut import_object = ImportObject::new(self.inner.clone());
                import_object.extend_with_pydict(py, imported_functions)?;

                self.inner.instantiate(&import_object.inner)
            } else {
                return Err(RuntimeError::py_err(
                    "The `imported_functions` parameter contains an unknown value. Python dictionaries or `wasmer.ImportObject` are the only supported values.".to_string()
                ));
            };

        // Instantiate the module.
        let instance = instance.map(Arc::new).map_err(|e| {
            RuntimeError::py_err(format!("Failed to instantiate the module:\n    {}", e))
        })?;

        let exports = instance.exports();

        // Collect the exported functions, globals and memory from the
        // WebAssembly module.
        let mut exported_functions = Vec::new();
        let mut exported_globals = Vec::new();
        let mut exported_memory = None;

        for (export_name, export) in exports {
            match export {
                Export::Function { .. } => exported_functions.push(export_name.clone()),
                Export::Global(global) => {
                    exported_globals.push((export_name.clone(), Arc::new(global.into())))
                }
                Export::Memory(memory) if exported_memory.is_none() => {
                    exported_memory = Some(Arc::new(memory.into()))
                }
                _ => (),
            }
        }

        // Instantiate the `Instance` Python class.
        Ok(Py::new(
            py,
            Instance::inner_new(
                instance.clone(),
                Py::new(
                    py,
                    ExportedFunctions {
                        instance: instance.clone(),
                        functions: exported_functions,
                    },
                )?,
                match exported_memory {
                    Some(memory) => Some(Py::new(py, Memory { memory })?),
                    None => None,
                },
                Py::new(
                    py,
                    ExportedGlobals {
                        globals: exported_globals,
                    },
                )?,
            ),
        )?)
    }

    /// The `exports` getter returns all the exported functions as a
    /// list of dictionaries with 2 pairs:
    ///
    ///   1. `"kind": <kind>`, where the kind is a `ExportKind` value.
    ///   2. `"name": <name>`, where the name is a string,
    #[getter]
    fn exports<'p>(&self, py: Python<'p>) -> PyResult<&'p PyList> {
        let exports = &self.inner.info().exports;
        let mut items: Vec<&PyDict> = Vec::with_capacity(exports.len());

        for (name, export_index) in exports.iter() {
            let dict = PyDict::new(py);

            dict.set_item(
                "kind",
                match export_index {
                    ExportIndex::Function(_) => ExportImportKind::Function,
                    ExportIndex::Memory(_) => ExportImportKind::Memory,
                    ExportIndex::Global(_) => ExportImportKind::Global,
                    ExportIndex::Table(_) => ExportImportKind::Table,
                },
            )?;
            dict.set_item("name", name)?;

            items.push(dict);
        }

        Ok(PyList::new(py, items))
    }

    /// The `imports` getter returns all the imported functions as a
    /// list of dictionaries with at least 3 pairs:
    ///
    ///   1. `"kind": <kind>`, where the kind is a `ImportKind` value.
    ///   2. `"namespace": <namespace>`, where the namespace is a string,
    ///   3. `"name": <name>`, where the name is a string.
    ///
    /// Additional pairs exist for the following kinds:
    ///
    ///   * `ImportKind.MEMORY` has the `"minimum_pages": {int}` and
    ///      `"maximum_pages": {int?}` pairs.
    ///   * `ImportKind.GLOBAL` has the `"mutable": {bool}` and
    ///     `"type": {string}` pairs.
    ///   * `ImportKind.TABLE` has the `"minimum_elements: {int}`,
    ///     `"maximum_elements: {int?}`, and `"element_type": {string}`
    ///     pairs.
    #[getter]
    fn imports<'p>(&self, py: Python<'p>) -> PyResult<&'p PyList> {
        let imports = self.inner.imports();
        let mut items: Vec<&PyDict> = Vec::with_capacity(imports.len());

        for import_descriptor in imports {
            let dict = PyDict::new(py);
            let module = import_descriptor.module();
            let name = import_descriptor.name();
            let ty = import_descriptor.ty();

            match ty {
                ExternDescriptor::Function(_) => {
                    dict.set_item("kind", ExportImportKind::Function as u8)?;
                    dict.set_item("namespace", module)?;
                    dict.set_item("name", name)?;
                }
                ExternDescriptor::Memory(memory) => {
                    dict.set_item("kind", ExportImportKind::Memory as u8)?;
                    dict.set_item("namespace", module)?;
                    dict.set_item("name", name)?;
                    dict.set_item("minimum_pages", memory.minimum.0)?;
                    dict.set_item(
                        "maximum_pages",
                        memory
                            .maximum
                            .map(|page| page.0.into_py(py))
                            .unwrap_or_else(|| py.None()),
                    )?;
                }
                ExternDescriptor::Global(global) => {
                    let mutable: bool = global.mutability.into();

                    dict.set_item("kind", ExportImportKind::Global as u8)?;
                    dict.set_item("namespace", module)?;
                    dict.set_item("name", name)?;
                    dict.set_item("mutable", mutable)?;
                    dict.set_item(
                        "type",
                        match global.ty {
                            Type::I32 => "i32",
                            Type::I64 => "i64",
                            Type::F32 => "f32",
                            Type::F64 => "f64",
                            Type::V128 => "v128",
                            Type::FuncRef => "funcref",
                            Type::ExternRef => "externref",
                        },
                    )?;
                }
                ExternDescriptor::Table(table) => {
                    dict.set_item("kind", ExportImportKind::Table as u8)?;
                    dict.set_item("namespace", module)?;
                    dict.set_item("name", name)?;
                    dict.set_item("minimum_elements", table.minimum)?;
                    dict.set_item(
                        "maximum_elements",
                        table
                            .maximum
                            .map(|number| number.into_py(py))
                            .unwrap_or_else(|| py.None()),
                    )?;
                    dict.set_item(
                        "element_type",
                        match table.ty {
                            Type::I32 => "i32",
                            Type::I64 => "i64",
                            Type::F32 => "f32",
                            Type::F64 => "f64",
                            Type::V128 => "v128",
                            Type::FuncRef => "funcref",
                            Type::ExternRef => "externref",
                        },
                    )?;
                }
            }

            items.push(dict);
        }

        Ok(PyList::new(py, items))
    }

    /// Read all the custom section names. To get the value of a
    /// custom section, use the `Module.custom_section()`
    /// function. This designed is motivated by saving memory.
    #[getter]
    fn custom_section_names<'p>(&self, py: Python<'p>) -> &'p PyList {
        PyList::new(py, self.inner.info().custom_sections.keys())
    }

    /// Read a specific custom section.
    #[text_signature = "($self, name, index=0)"]
    #[args(index = 0)]
    fn custom_section<'p>(&self, py: Python<'p>, name: String, index: usize) -> PyObject {
        match self.inner.info().custom_sections(&name).nth(index) {
            Some(bytes) => PyBytes::new(py, &bytes).into_py(py),
            None => py.None(),
        }
    }

    /// Serialize the module into Python bytes.
    ///
    /// # Examples
    ///
    /// ```py
    /// module1 = Module(wasm_bytes)
    /// serialized_module = module1.serialize()
    /// del module1
    ///
    /// module2 = Module.deserialize(serialized_module)
    /// ```
    #[text_signature = "($self)"]
    fn serialize<'p>(&self, py: Python<'p>) -> PyResult<&'p PyBytes> {
        // Get the module artifact.
        match self.inner.cache() {
            // Serialize the artifact.
            Ok(artifact) => match artifact.serialize() {
                Ok(serialized_artifact) => Ok(PyBytes::new(py, serialized_artifact.as_slice())),
                Err(_) => Err(RuntimeError::py_err(
                    "Failed to serialize the module artifact.",
                )),
            },
            Err(_) => Err(RuntimeError::py_err("Failed to get the module artifact.")),
        }
    }

    /// Deserialize Python bytes into a module instance.
    ///
    /// See `Module.serialize` to get an example.
    #[text_signature = "(bytes)"]
    #[staticmethod]
    fn deserialize(bytes: &PyAny, py: Python) -> PyResult<Py<Module>> {
        // Read the bytes.
        let serialized_module = bytes.downcast::<PyBytes>()?.as_bytes();

        // Deserialize the artifact.
        match unsafe { Artifact::deserialize(serialized_module) } {
            Ok(artifact) => {
                // Get the module from the artifact.
                match runtime::load_cache_with(artifact) {
                    Ok(module) => Ok(Py::new(
                        py,
                        Self {
                            inner: Arc::new(module),
                        },
                    )?),
                    Err(_) => Err(RuntimeError::py_err(
                        "Failed to compile the serialized module.",
                    )),
                }
            }
            Err(_) => Err(RuntimeError::py_err("Failed to deserialize the module.")),
        }
    }

    /// Generates a fresh `ImportObject` object.
    ///
    /// # Examples
    ///
    /// ```py
    /// module = Module(wasm_bytes)
    /// import_object = module.generate_import_object()
    /// # do something with `import_object`
    /// instance = module.instantiate(import_object)
    /// ```
    #[text_signature = "($self)"]
    fn generate_import_object(&self) -> ImportObject {
        ImportObject::new(self.inner.clone())
    }

    /// Checks whether the module contains WASI definitions.
    ///
    /// # Exampels
    ///
    /// ```py
    /// module = Module(wasm_bytes)
    /// is_wasi = module.is_wasi_module
    /// ```
    #[getter]
    fn is_wasi_module(&self) -> bool {
        wasmer_wasi::is_wasi_module(&self.inner.into_inner())
    }

    /// Checks the WASI version if any.
    ///
    /// A strict detection expects that all imports live in a single
    /// WASI namespace. A non-strict detection (the default) expects
    /// that at least one WASI namespace exists to detect the
    /// version. Note that the strict detection is faster than the
    /// non-strict one.
    ///
    /// # Examples
    ///
    /// ```py
    /// module = Module(wasm_bytes)
    /// wasi_version = module.wasi_version(strict=True)
    ///
    /// assert wasi_version == WasmVersion.Snapshot1
    /// ```
    #[text_signature = "($self, strict=False)"]
    #[args(strict = false)]
    fn wasi_version<'p>(&self, py: Python<'p>, strict: bool) -> PyObject {
        let version: Option<wasi::Version> =
            wasmer_wasi::get_wasi_version(&self.inner.into_inner(), strict).map(Into::into);

        match version {
            Some(version) => version.to_object(py),
            None => py.None(),
        }
    }
}
