
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo, PyDowncastError, AsPyPointer, PyNumberProtocol};
use pyo3::PyResult;
use std::iter::FromIterator;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims, DataType};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use pyo3::exceptions::PyValueError;
use serde::{Serialize, Deserialize};


macro_rules! impl_save_load {
    ($class_name:ident, $inner_field:ident) => {
        #[pymethods]
        impl $class_name {
            #[text_signature = "(file)"]
            pub fn save(&self, file: String) -> PyResult<()> {
                pickle(&self.$inner_field, file)
            }
            #[text_signature = "()"]
            pub fn clone(&self) -> Self {
                Self{$inner_field:self.$inner_field.clone()}
            }
            #[staticmethod]
            #[text_signature = "(file)"]
            pub fn load(file: String) -> PyResult<Self> {
                unpickle(file).map(|s| Self { $inner_field: s })
            }
        }
    };
}

pub(crate) use impl_save_load;

pub fn arr3<'py, T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject) -> PyResult<[T; 3]> {
    Ok(if let Ok(t) = t.extract::<(T, T, T)>(py) {
        [t.0, t.1, t.2]
    } else if let Ok(t) = t.extract::<Vec<T>>(py) {
        [t[0], t[1], t[2]]
    } else {
        let array = py_any_as_numpy(t.as_ref(py))?;
        let t = unsafe { array.as_slice()? };
        [t[0], t[1], t[2]]
    })
}

pub fn arrX<'py,T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject, default_x: T, default_y: T, default_z: T) -> PyResult<[T; 3]> {
    Ok(if t.is_none(py) {
        [default_x, default_y, default_z]
    } else if let Ok(t) = t.extract::<T>(py) {
        [t, default_y, default_z]
    } else if let Ok(t) = t.extract::<(T, T)>(py) {
        [t.0, t.1, default_z]
    } else if let Ok(t) = t.extract::<(T, T, T)>(py) {
        [t.0, t.1, t.2]
    } else {
        let d = [default_x, default_y, default_z];
        fn to3<T:Element+Copy>(arr: &[T], mut d: [T; 3]) -> [T; 3] {
            for i in 0..arr.len().min(3) {
                d[i] = arr[i];
            }
            d
        }
        if let Ok(t) = t.extract::<Vec<T>>(py) {
            to3(&t, d)
        } else {
            let array = py_any_as_numpy(t.as_ref(py))?;
            let t = unsafe { array.as_slice()? };
            to3(&t, d)
        }
    })
}

pub fn arr2<'py,T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject) -> PyResult<[T; 2]> {
    Ok(if let Ok(t) = t.extract::<(T, T)>(py) {
        [t.0, t.1]
    } else if let Ok(t) = t.extract::<Vec<T>>(py) {
        [t[0], t[1]]
    } else {
        let array = py_any_as_numpy(t.as_ref(py))?;
        let t = unsafe { array.as_slice()? };
        [t[0], t[1]]
    })
}

pub fn pickle<T: Serialize>(val: &T, file: String) -> PyResult<()> {
    let o = OpenOptions::new()
        .write(true)
        .create(true)
        .open(file)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    ciborium::ser::into_writer(val,&mut BufWriter::new(o)).map_err(|err|PyValueError::new_err(err.to_string()))
}

pub fn ocl_err_to_py_ex(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}
pub fn unpickle<T:Deserialize<'static>>(file: String) -> PyResult<T> {
    let o = OpenOptions::new()
        .read(true)
        .open(file)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    ciborium::de::from_reader(&mut BufReader::new(o)).map_err(|err|PyValueError::new_err(err.to_string()))
}
pub fn py_any_as_numpy<T:Element>(input: &PyAny) -> Result<&PyArrayDyn<T>, PyErr> {
    let array = unsafe {
        if npyffi::PyArray_Check(input.as_ptr()) == 0 {
            return Err(PyDowncastError::new(input, "PyArray<T, D>").into());
        }
        &*(input as *const PyAny as *const PyArrayDyn<u8>)
    };
    if !array.is_c_contiguous(){
        return Err(PyValueError::new_err("Numpy array is not C contiguous"));
    }
    let actual_dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
    if T::DATA_TYPE != actual_dtype {
        return Err(PyValueError::new_err(format!("Expected numpy array of dtype {:?} but got {:?}", T::DATA_TYPE, actual_dtype)));
    }
    let array = unsafe { &*(input as *const PyAny as *const PyArrayDyn<T>) };
    Ok(array)
}
