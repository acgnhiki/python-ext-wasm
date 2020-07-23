//! This module contains a helper to build an `ImportObject` and build
//! the host function logic.

use crate::{
    instance::exports::ExportImportKind,
    wasi,
    wasmer::{core, runtime, wasi as wasmer_wasi},
};
use pyo3::{
    exceptions::RuntimeError,
    prelude::*,
    types::{PyDict, PyList},
    PyObject,
};
use std::sync::Arc;

#[pyclass(unsendable)]
/// `ImportObject` is a Python class that represents the
/// `crate::wasmer::core::import::ImportObject`.
pub struct ImportObject {
    pub(crate) inner: runtime::ImportObject,

    #[allow(unused)]
    pub(crate) module: Arc<runtime::Module>,

    /// This field is unused as is, but is required to keep a
    /// reference to host function `PyObject`.
    #[allow(unused)]
    pub(crate) host_function_references: Vec<PyObject>,
}

impl ImportObject {
    pub fn new(module: Arc<runtime::Module>) -> Self {
        Self {
            inner: runtime::ImportObject::new(),
            module,
            host_function_references: Vec::new(),
        }
    }

    pub fn new_with_wasi(
        module: Arc<runtime::Module>,
        version: wasi::Version,
        wasi: &mut wasi::Wasi,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: wasmer_wasi::generate_import_object_from_env(
                &core::get_global_store(),
                wasmer_wasi::WasiEnv::new(wasi.inner.build().map_err(|error| {
                    RuntimeError::py_err(format!("Failed to create the WASI state: {}", error))
                })?),
                version.into(),
            ),
            module,
            host_function_references: Vec::new(),
        })
    }

    #[cfg(not(all(unix, target_arch = "x86_64")))]
    pub fn extend_with_pydict(&mut self, _py: Python, imported_functions: &PyDict) -> PyResult<()> {
        if imported_functions.is_empty() {
            Ok(())
        } else {
            Err(RuntimeError::py_err(
                "Imported functions are not yet supported for this platform and architecture.",
            ))
        }
    }

    #[cfg(all(unix, target_arch = "x86_64"))]
    pub fn extend_with_pydict<'py>(
        &mut self,
        py: Python,
        imported_functions: &PyDict,
    ) -> PyResult<()> {
        use crate::wasmer::{
            core::{import::Namespace, typed_func::DynamicFunc, types::ExternDescriptor},
            runtime::{
                types::{FuncSig, Type},
                Value,
            },
        };
        use pyo3::{
            types::{PyFloat, PyLong, PyString, PyTuple},
            AsPyPointer,
        };
        use std::collections::HashMap;

        let imports = self.module.imports();
        let import_descriptors: HashMap<(&str, &str), &FuncSig> = imports
            .iter()
            .filter_map(|import_type| {
                Some((
                    (import_type.module(), import_type.name()),
                    match import_type.ty() {
                        ExternDescriptor::Function(function_type) => function_type,
                        _ => return None,
                    },
                ))
            })
            .collect();

        let mut host_function_references = Vec::with_capacity(imported_functions.len());

        for (namespace_name, namespace) in imported_functions.iter() {
            let namespace_name = namespace_name
                .downcast::<PyString>()
                .map_err(|_| RuntimeError::py_err("Namespace name must be a string.".to_string()))?
                .to_string()?;

            let mut import_namespace = Namespace::new();

            for (function_name, function) in namespace
                .downcast::<PyDict>()
                .map_err(|_| RuntimeError::py_err("Namespace must be a dictionary.".to_string()))?
                .into_iter()
            {
                let function_name = function_name
                    .downcast::<PyString>()
                    .map_err(|_| {
                        RuntimeError::py_err("Function name must be a string.".to_string())
                    })?
                    .to_string()?;

                if !function.is_callable() {
                    return Err(RuntimeError::py_err(format!(
                        "Function for `{}` is not callable.",
                        function_name
                    )));
                }

                let imported_function_signature = import_descriptors
                    .get(&(&namespace_name, &function_name))
                    .ok_or_else(|| {
                        RuntimeError::py_err(
                            format!(
                                "The imported function `{}.{}` does not have a signature in the WebAssembly module.",
                                namespace_name,
                                function_name
                            )
                        )
                    })?;

                let mut input_types = vec![];
                let mut output_types = vec![];

                if !function.hasattr("__annotations__")? {
                    return Err(RuntimeError::py_err(format!(
                        "Function `{}` must have type annotations for parameters and results.",
                        function_name
                    )));
                }

                let annotations = function
                    .getattr("__annotations__")?
                    .downcast::<PyDict>()
                    .map_err(|_| {
                        RuntimeError::py_err(format!(
                            "Failed to read annotations of function `{}`.",
                            function_name
                        ))
                    })?;

                if annotations.len() > 0 {
                    for ((annotation_name, annotation_value), expected_type) in
                        annotations.iter().zip(
                            imported_function_signature
                                .params()
                                .iter()
                                .chain(imported_function_signature.results().iter()),
                        )
                    {
                        let ty = match annotation_value.to_string().as_str() {
                            "i32" | "I32" | "<class 'int'>" if expected_type == &Type::I32 => Type::I32,
                            "i64" | "I64" | "<class 'int'>" if expected_type == &Type::I64 => Type::I64,
                            "f32" | "F32" | "<class 'float'>" if expected_type == &Type::F32 => Type::F32,
                            "f64" | "F64" | "<class 'float'>" if expected_type == &Type::F64 => Type::F64,
                            t @ _ => {
                                return Err(RuntimeError::py_err(format!(
                                    "Type `{}` is not a supported type, or is not the expected type (`{}`).",
                                    t, expected_type
                                )))
                            }
                        };

                        match annotation_name.to_string().as_str() {
                            "return" => output_types.push(ty),
                            _ => input_types.push(ty),
                        }
                    }

                    if output_types.len() > 1 {
                        return Err(RuntimeError::py_err(
                            "Function must return only one type, many given.".to_string(),
                        ));
                    }
                } else {
                    input_types.extend(imported_function_signature.params());
                    output_types.extend(imported_function_signature.results());
                }

                let function = function.to_object(py);

                host_function_references.push(function.clone_ref(py));

                let function_implementation = DynamicFunc::new(
                    &FuncSig::new(input_types, output_types.clone()),
                    move |_, inputs: &[Value]| -> Result<Vec<Value>, core::error::RuntimeError> {
                        let gil = GILGuard::acquire();
                        let py = gil.python();

                        let inputs: Vec<PyObject> = inputs
                            .iter()
                            .map(|input| {
                                Ok(match input {
                                    Value::I32(value) => value.to_object(py),
                                    Value::I64(value) => value.to_object(py),
                                    Value::F32(value) => value.to_object(py),
                                    Value::F64(value) => value.to_object(py),
                                    Value::V128(value) => value.to_object(py),
                                    input => {
                                        return Err(core::error::RuntimeError::new(format!(
                                            "Input `{:?}` isn't supported.",
                                            input
                                        )));
                                    }
                                })
                            })
                            .collect::<Result<_, _>>()?;

                        if function.as_ptr().is_null() {
                            return Err(core::error::RuntimeError::new(
                                "Host function implementation is null. Maybe it has moved?",
                            ));
                        }

                        let results =
                            function
                                .call(py, PyTuple::new(py, inputs), None)
                                .map_err(|_| {
                                    core::error::RuntimeError::new(
                                        "Failed to call the host function.",
                                    )
                                })?;

                        let results = match results.cast_as::<PyTuple>(py) {
                            Ok(results) => results,
                            Err(_) => PyTuple::new(py, vec![results]),
                        };

                        let outputs: Vec<Value> = results
                            .iter()
                            .zip(output_types.iter())
                            .map(|(result, output)| {
                                Ok(match output {
                                    Type::I32 => Value::I32(
                                        result
                                            .downcast::<PyLong>()
                                            .unwrap()
                                            .extract::<i32>()
                                            .unwrap(),
                                    ),
                                    Type::I64 => Value::I64(
                                        result
                                            .downcast::<PyLong>()
                                            .unwrap()
                                            .extract::<i64>()
                                            .unwrap(),
                                    ),
                                    Type::F32 => Value::F32(
                                        result
                                            .downcast::<PyFloat>()
                                            .unwrap()
                                            .extract::<f32>()
                                            .unwrap(),
                                    ),
                                    Type::F64 => Value::F64(
                                        result
                                            .downcast::<PyFloat>()
                                            .unwrap()
                                            .extract::<f64>()
                                            .unwrap(),
                                    ),
                                    Type::V128 => Value::V128(
                                        result
                                            .downcast::<PyLong>()
                                            .unwrap()
                                            .extract::<u128>()
                                            .unwrap(),
                                    ),
                                    output => {
                                        return Err(core::error::RuntimeError::new(format!(
                                            "Output `{:?}` isn't supported.",
                                            output
                                        )));
                                    }
                                })
                            })
                            .collect::<Result<_, _>>()?;

                        Ok(outputs)
                    },
                );

                import_namespace.insert(function_name, function_implementation);
            }

            self.inner.register(namespace_name, import_namespace);
        }

        self.host_function_references = host_function_references;

        Ok(())
    }
}

#[pymethods]
/// Implement methods on the `ImportObject` Python class.
impl ImportObject {
    /// Extend the `ImportObject` by adding host functions stored in a Python directory.
    ///
    /// # Examples
    ///
    /// ```py
    /// # Our host function.
    /// def sum(x: int, y: int) -> int:
    ///     return x + y
    ///
    /// module = Module(wasm_bytes)
    ///
    /// # Generate an import object for this module.
    /// import_object = module.generate_import_object()
    ///
    /// # Register the `env.sum` host function.
    /// import_object.extend({
    ///     "env": {
    ///         "sum": sum
    ///     }
    /// })
    ///
    /// # Ready to instantiate the module.
    /// instance = module.instantiate(import_object)
    /// ```
    #[text_signature = "($self, imported_functions)"]
    pub fn extend(&mut self, py: Python, imported_functions: &PyDict) -> PyResult<()> {
        self.extend_with_pydict(py, imported_functions)
    }

    /// Read the descriptors of the imports.
    ///
    /// A descriptor for an import a dictionary with the following
    /// entries:
    ///
    ///   1. `kind` of type `ImportKind`, to represent the kind of
    ///      imported entity,
    ///   2. `namespace` of type `String`, to represent the namespace
    ///      of the imported entity,
    ///   3. `name` of type `String`, to represent the name of the
    ///      imported entity.
    #[text_signature = "($self)"]
    pub fn import_descriptors<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        let iterator = self.inner.clone().into_iter();
        let mut items: Vec<&PyDict> = Vec::with_capacity(iterator.size_hint().0);

        for ((namespace, name), import) in iterator {
            let dict = PyDict::new(py);

            let import: crate::wasmer::core::export::RuntimeExport = import;

            dict.set_item("kind", ExportImportKind::from(&import) as u8)?;
            dict.set_item("namespace", namespace)?;
            dict.set_item("name", name)?;

            items.push(dict);
        }

        Ok(PyList::new(py, items))
    }
}
