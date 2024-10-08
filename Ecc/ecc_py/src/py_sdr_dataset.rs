// use pyo3::prelude::*;
// use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo, PyDowncastError, AsPyPointer, PyNumberProtocol};
//
// ///
// /// CpuSdrDataset(shape, sdr)
// ///
// ///
// #[pyclass]
// pub struct CpuSdrDataset {
//     pub(crate) sdr: htm::CpuSdrDataset,
// }
//
// #[pyclass]
// pub struct SubregionIndices {
//     pub(crate) sdr: htm::SubregionIndices,
// }
//
// #[pymethods]
// impl SubregionIndices {
//     #[getter]
//     fn get_in_shape(&self) -> Vec<Idx> {
//         self.sdr.shape().to_vec()
//     }
//     #[getter]
//     fn get_dataset_len(&self) -> usize {
//         self.sdr.dataset_len()
//     }
//     #[text_signature = "(idx)"]
//     fn get_sample_idx(&self, idx: usize) -> u32 {
//         self.sdr[idx].channels()
//     }
//     #[text_signature = "(idx)"]
//     fn get_output_column_pos(&self, idx: usize) -> Vec<Idx> {
//         self.sdr[idx].grid().to_vec()
//     }
// }
//
// #[pyclass]
// pub struct Occurrences {
//     pub(crate) c: htm::Occurrences,
// }
//
// #[pymethods]
// impl Occurrences {
//     #[text_signature = "(input_idx,label)"]
//     pub fn prob(&self, input_idx: usize, label: usize) -> f32 {
//         self.c.prob(input_idx, label)
//     }
//     #[text_signature = "()"]
//     pub fn normalise_wrt_labels(&mut self) {
//         self.c.normalise_wrt_labels()
//     }
//     #[text_signature = "()"]
//     pub fn normalise_class_probs(&mut self) {
//         self.c.normalise_class_probs()
//     }
//     #[text_signature = "()"]
//     pub fn log(&mut self) {
//         self.c.log()
//     }
//     #[text_signature = "()"]
//     pub fn occurrences(&self) -> Vec<f32> {
//         self.c.occurrences().to_vec()
//     }
//     #[text_signature = "()"]
//     pub fn square_weights(&mut self) {
//         self.c.sqrt_weights()
//     }
//     #[text_signature = "()"]
//     pub fn sqrt_weights(&mut self) {
//         self.c.sqrt_weights()
//     }
//     #[text_signature = "()"]
//     pub fn exp_weights(&mut self) {
//         self.c.exp_weights()
//     }
//     #[text_signature = "()"]
//     pub fn log_weights(&mut self) {
//         self.c.log_weights()
//     }
//     #[text_signature = "()"]
//     pub fn square_class_probs(&mut self) {
//         self.c.sqrt_class_probs()
//     }
//     #[text_signature = "()"]
//     pub fn sqrt_class_probs(&mut self) {
//         self.c.sqrt_class_probs()
//     }
//     #[text_signature = "()"]
//     pub fn exp_class_probs(&mut self) {
//         self.c.exp_class_probs()
//     }
//     #[text_signature = "()"]
//     pub fn log_class_probs(&mut self) {
//         self.c.log_class_probs()
//     }
//     #[text_signature = "(label)"]
//     pub fn occurrences_for_label(&self, lbl: usize) -> Vec<f32> {
//         self.c.occurrences_for_label(lbl).to_vec()
//     }
//     #[text_signature = "(sdr)"]
//     pub fn collect_votes_per_column_and_lbl(&self, sdr: &CpuSDR) -> Tensor {
//         Tensor { ecc: self.c.collect_votes_per_column_and_lbl(&sdr.sdr) }
//     }
//     #[text_signature = "()"]
//     pub fn aggregate_invariant_to_column(&mut self) {
//         self.c.aggregate_invariant_to_column()
//     }
//     #[text_signature = "(sdr,min_deviation_from_mean)"]
//     pub fn classify_and_count_votes_per_lbl(&self, sdr: &CpuSDR, min_deviation_from_mean: f32) -> Vec<u32> {
//         self.c.classify_and_count_votes_per_lbl(&sdr.sdr, min_deviation_from_mean)
//     }
//     #[text_signature = "(sdr,min_deviation_from_mean)"]
//     pub fn classify_per_column(&self, sdr: &CpuSDR, min_deviation_from_mean: f32) -> Vec<isize> {
//         self.c.classify_per_column(&sdr.sdr, min_deviation_from_mean).into_vec()
//     }
//     #[text_signature = "(sdr)"]
//     pub fn compute_class_sums(&mut self) {
//         self.c.compute_class_sums()
//     }
//     #[text_signature = "(sdr)"]
//     pub fn compute_class_probs(&mut self) {
//         self.c.compute_class_probs()
//     }
//     #[text_signature = "(class_idx)"]
//     fn class_prob_of(&self, class_idx: usize) -> f32 {
//         self.c.class_prob()[class_idx]
//     }
//     #[getter]
//     fn class_prob(&self) -> Vec<f32> {
//         self.c.class_prob().to_vec()
//     }
//     #[getter]
//     fn num_classes(&self) -> usize {
//         self.c.num_classes()
//     }
//     #[getter]
//     fn get_shape(&self) -> Vec<Idx> {
//         self.c.shape().to_vec()
//     }
//     #[text_signature = "()"]
//     fn clear_class_prob(&mut self) {
//         self.c.clear_class_prob()
//     }
//     #[text_signature = "(sdr,min_deviation_from_mean)"]
//     fn classify(&self, sdr: &CpuSDR, min_deviation_from_mean: Option<f32>) -> usize {
//         if let Some(min_deviation_from_mean) = min_deviation_from_mean {
//             self.c.classify_with_most_votes(&sdr.sdr, min_deviation_from_mean)
//         } else {
//             self.c.classify(&sdr.sdr)
//         }
//     }
//     #[text_signature = "(sdr_dataset, min_deviation_from_mean)"]
//     fn batch_classify<'py>(&self, py: Python<'py>, sdr: &CpuSdrDataset, min_deviation_from_mean: Option<f32>) -> &'py PyArray<u32, Ix1> {
//         let v = if let Some(min_deviation_from_mean) = min_deviation_from_mean {
//             self.c.batch_classify_invariant_to_column(&sdr.sdr, min_deviation_from_mean)
//         } else {
//             self.c.batch_classify(&sdr.sdr)
//         };
//         PyArray::from_vec(py, v)
//     }
// }
//
// impl CpuSdrDataset {
//     fn to_numpy_<'py, T: Element + Copy>(&self, py: Python<'py>, idx: usize, one: T) -> &'py PyArray<T, Ix3> {
//         let sdr = &self.sdr[idx];
//         let mut arr = PyArray3::zeros(py, self.sdr.shape().map(Idx::as_usize), false);
//         let s = unsafe { arr.as_slice_mut().unwrap() };
//         sdr.iter().for_each(|&i| s[i.as_usize()] = one);
//         arr
//     }
// }
//
// #[pymethods]
// impl CpuSdrDataset {
//     #[new]
//     pub fn new(shape: [Idx; 3], sdr: Option<Vec<PyRef<CpuSDR>>>) -> Self {
//         Self {
//             sdr: if let Some(s) = sdr {
//                 let mut d = htm::CpuSdrDataset::with_capacity(s.len(), shape);
//                 d.extend(s.iter().map(|p| p.sdr.clone()));
//                 d
//             } else {
//                 htm::CpuSdrDataset::new(shape)
//             }
//         }
//     }
//     #[text_signature = "(idx)"]
//     fn to_numpy<'py>(&self, py: Python<'py>, idx: usize) -> &'py PyArray<u32, Ix3> {
//         self.to_numpy_(py, idx, 1)
//     }
//     #[text_signature = "(idx)"]
//     fn to_bool_numpy<'py>(&self, py: Python<'py>, idx: usize) -> &'py PyArray<bool, Ix3> {
//         self.to_numpy_(py, idx, true)
//     }
//     #[text_signature = "(idx)"]
//     fn to_f32_numpy<'py>(&self, py: Python<'py>, idx: usize) -> &'py PyArray<f32, Ix3> {
//         self.to_numpy_(py, idx, 1.)
//     }
//     #[text_signature = "(idx)"]
//     fn to_f64_numpy<'py>(&self, py: Python<'py>, idx: usize) -> &'py PyArray<f64, Ix3> {
//         self.to_numpy_(py, idx, 1.)
//     }
//     #[getter]
//     fn get_shape(&self) -> Vec<Idx> {
//         self.sdr.shape().to_vec()
//     }
//     #[getter]
//     fn get_grid(&self) -> Vec<Idx> {
//         self.sdr.shape().grid().to_vec()
//     }
//     #[getter]
//     fn get_volume(&self) -> Idx {
//         self.sdr.shape().size()
//     }
//     #[getter]
//     fn get_channels(&self) -> Idx {
//         self.sdr.shape().channels()
//     }
//     #[getter]
//     fn get_width(&self) -> Idx {
//         self.sdr.shape().width()
//     }
//     #[getter]
//     fn get_height(&self) -> Idx {
//         self.sdr.shape().height()
//     }
//     #[getter]
//     fn get_area(&self) -> Idx {
//         self.sdr.shape().grid().product()
//     }
//     #[text_signature = "(min_cardinality)"]
//     pub fn filter_by_cardinality_threshold(&mut self, min_cardinality: Idx) {
//         self.sdr.filter_by_cardinality_threshold(min_cardinality)
//     }
//     // #[text_signature = "(number_of_samples,ecc,log)"]
//     // fn train_machine_with_patches(&self, number_of_samples: usize, ecc: &mut EccNet,log:Option<usize>){
//     //     if let Some(log) = log {
//     //         assert!(log > 0, "Logging interval must be greater than 0");
//     //         self.sdr.train_machine_with_patches(number_of_samples, &mut ecc.ecc, &mut rand::thread_rng(), |i| if i % log == 0 { println!("Processed samples {}", i + 1) })
//     //     } else {
//     //         self.sdr.train_machine_with_patches(number_of_samples, &mut ecc.ecc, &mut rand::thread_rng(), |i| {})
//     //     }
//     // }
//     #[text_signature = "(number_of_samples,drift,patches_per_sample,ecc,log)"]
//     pub fn train_with_patches(&self, number_of_samples: usize, drift: [Idx; 2], patches_per_sample: usize, ecc: &mut EccLayer, log: Option<usize>) {
//         if let Some(log) = log {
//             assert!(log > 0, "Logging interval must be greater than 0");
//             self.sdr.train_with_patches(number_of_samples, drift, patches_per_sample, &mut ecc.ecc, &mut rand::thread_rng(), |i| if i % log == 0 { println!("Processed samples {}", i + 1) })
//         } else {
//             self.sdr.train_with_patches(number_of_samples, drift, patches_per_sample, &mut ecc.ecc, &mut rand::thread_rng(), |i| {})
//         }
//     }
//     #[text_signature = "(number_of_samples,ecc,log)"]
//     pub fn train(&self, number_of_samples: usize, ecc: &mut EccLayer, log: Option<usize>) {
//         if let Some(log) = log {
//             self.sdr.train(number_of_samples, &mut ecc.ecc, &mut rand::thread_rng(), |i| if i % log == 0 { println!("Processed samples {}", i + 1) })
//         } else {
//             self.sdr.train(number_of_samples, &mut ecc.ecc, &mut rand::thread_rng(), |i| {})
//         }
//     }
//     #[text_signature = "()"]
//     fn clear(&mut self) {
//         self.sdr.clear()
//     }
//     #[text_signature = "(sdr)"]
//     fn push(&mut self, sdr: &CpuSDR) {
//         self.sdr.push(sdr.sdr.clone())
//     }
//     #[text_signature = "()"]
//     fn pop(&mut self) -> Option<CpuSDR> {
//         self.sdr.pop().map(|sdr| CpuSDR { sdr })
//     }
//     #[text_signature = "()"]
//     fn rand(&self) -> Option<CpuSDR> {
//         self.sdr.rand(&mut rand::thread_rng()).map(|sdr| CpuSDR { sdr: sdr.clone() })
//     }
//     #[text_signature = "(ecc,indices)"]
//     fn conv_subregion_indices_with_ecc(&self, ecc: &EccLayer, indices: &SubregionIndices) -> Self {
//         Self { sdr: self.sdr.conv_subregion_indices_with_ecc(&ecc.ecc, &indices.sdr) }
//     }
//     #[text_signature = "(ecc,indices)"]
//     fn conv_subregion_indices_with_net(&self, ecc: &EccNet, indices: &SubregionIndices) -> Self {
//         Self { sdr: self.sdr.conv_subregion_indices_with_net(&ecc.ecc, &indices.sdr) }
//     }
//     #[text_signature = "(kernel,stride,indices)"]
//     fn conv_subregion_indices_with_ker(&self, kernel: [Idx; 2], stride: [Idx; 2], indices: &SubregionIndices) -> Self {
//         Self { sdr: self.sdr.conv_subregion_indices_with_ker(kernel, stride, &indices.sdr) }
//     }
//     #[text_signature = "(number_of_samples,original_dataset)"]
//     pub fn extend_from_rand_subregions(&mut self, number_of_samples: usize, original: &CpuSdrDataset) {
//         self.sdr.extend_from_rand_subregions(number_of_samples, &original.sdr, &mut rand::thread_rng())
//     }
//     #[text_signature = "(kernel,stride,number_of_samples,original_dataset)"]
//     fn extend_from_conv_rand_subregion(&mut self, kernel: [Idx; 2], stride: [Idx; 2], number_of_samples: usize, original: &CpuSdrDataset) {
//         let conv = ConvShape::new_in(*original.sdr.shape(), 1, kernel, stride);
//         self.sdr.extend_from_conv_rand_subregion(&conv, number_of_samples, &original.sdr, &mut rand::thread_rng())
//     }
//     #[text_signature = "(kernel,stride,indices,original_dataset)"]
//     fn extend_from_conv_subregion_indices(&mut self, kernel: [Idx; 2], stride: [Idx; 2], indices: &SubregionIndices, original: &CpuSdrDataset) {
//         let conv = ConvShape::new_in(*original.sdr.shape(), 1, kernel, stride);
//         self.sdr.extend_from_conv_subregion_indices(&conv, &indices.sdr, &original.sdr)
//     }
//     #[text_signature = "(patch_size,number_of_samples)"]
//     fn gen_rand_2d_patches(&self, patch_size: [Idx; 2], number_of_samples: usize) -> Self {
//         Self { sdr: self.sdr.gen_rand_2d_patches(patch_size, number_of_samples, &mut rand::thread_rng()) }
//     }
//     #[text_signature = "(subregion_size,number_of_samples)"]
//     fn gen_rand_subregions(&self, subregion: [Idx; 3], number_of_samples: usize) -> Self {
//         Self { sdr: self.sdr.gen_rand_subregions(subregion, number_of_samples, &mut rand::thread_rng()) }
//     }
//     #[text_signature = "(out_shape,number_of_samples)"]
//     fn gen_rand_conv_subregion_indices(&self, out_shape: [Idx; 2], number_of_samples: usize) -> SubregionIndices {
//         SubregionIndices { sdr: self.sdr.gen_rand_conv_subregion_indices(out_shape, number_of_samples, &mut rand::thread_rng()) }
//     }
//     #[text_signature = "(kernel,stride,number_of_samples)"]
//     fn gen_rand_conv_subregion_indices_with_ker(&self, kernel: [Idx; 2], stride: [Idx; 2], number_of_samples: usize) -> SubregionIndices {
//         SubregionIndices { sdr: self.sdr.gen_rand_conv_subregion_indices_with_ker(&kernel, &stride, number_of_samples, &mut rand::thread_rng()) }
//     }
//     #[text_signature = "(ecc_dense,number_of_samples)"]
//     fn gen_rand_conv_subregion_indices_with_ecc(&self, ecc: &EccLayer, number_of_samples: usize) -> SubregionIndices {
//         SubregionIndices { sdr: self.sdr.gen_rand_conv_subregion_indices_with_ecc(&ecc.ecc, number_of_samples, &mut rand::thread_rng()) }
//     }
//     #[text_signature = "(ecc_net,number_of_samples)"]
//     fn gen_rand_conv_subregion_indices_with_net(&self, ecc: &EccNet, number_of_samples: usize) -> SubregionIndices {
//         SubregionIndices { sdr: self.sdr.gen_rand_conv_subregion_indices_with_net(&ecc.ecc, number_of_samples, &mut rand::thread_rng()) }
//     }
//
//     #[text_signature = "(labels,number_of_classes)"]
//     fn count_per_label(&self, labels: &PyAny, number_of_classes: usize) -> PyResult<Occurrences> {
//         let array = unsafe {
//             if npyffi::PyArray_Check(labels.as_ptr()) == 0 {
//                 return Err(PyDowncastError::new(labels, "PyArray<T, D>").into());
//             }
//             &*(labels as *const PyAny as *const PyArrayDyn<u8>)
//         };
//         if !array.is_c_contiguous() {
//             return Err(PyValueError::new_err("Numpy array is not C contiguous"));
//         }
//         let dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
//         fn f<T: Element>(sdr: &htm::CpuSdrDataset, labels: &PyAny, number_of_classes: usize, f: impl Fn(&T) -> usize) -> PyResult<Occurrences> {
//             let labels = unsafe { &*(labels as *const PyAny as *const PyArrayDyn<T>) };
//             let labels = unsafe { labels.as_slice()? };
//             Ok(Occurrences { c: sdr.count_per_label(labels, number_of_classes, f) })
//         }
//         match dtype {
//             u8::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &u8| *f as usize),
//             u32::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &u32| *f as usize),
//             u64::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &u64| *f as usize),
//             i8::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &i8| *f as usize),
//             i32::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &i32| *f as usize),
//             i64::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &i64| *f as usize),
//             usize::DATA_TYPE => f(&self.sdr, labels, number_of_classes, |f: &usize| *f),
//             d => Err(PyValueError::new_err(format!("Unexpected dtype {:?} of numpy array ", d)))
//         }
//     }
//     #[text_signature = "()"]
//     fn count<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray<u32, Ix3>> {
//         let v = self.sdr.count();
//         let a = PyArray::from_vec(py, v);
//         a.reshape(self.sdr.shape().map(Idx::as_usize))
//     }
//     // #[text_signature = "(outputs)"]
//     // fn measure_receptive_fields<'py>(&self, py: Python<'py>, outputs: &CpuSdrDataset) -> PyResult<&'py PyArray<u32, Ix4>> {
//     //     let ov = outputs.sdr.shape().size();
//     //     let v = self.sdr.measure_receptive_fields(&outputs.sdr);
//     //     let a = PyArray::from_vec(py, v);
//     //     let s = self.sdr.shape();
//     //     let s = [s[0], s[1], s[2], ov].map(Idx::as_usize);
//     //     a.reshape(s)
//     // }
//     // #[text_signature = "(ecc_dense)"]
//     // fn batch_infer(&self, ecc: &EccLayer) -> CpuSdrDataset {
//     //     Self { sdr: self.sdr.batch_infer(&ecc.ecc) }
//     // }
//     #[text_signature = "(ecc_net,learn)"]
//     fn net_infer(&self, ecc: &mut EccNet,learn:Option<bool>) -> CpuSdrDataset {
//         Self { sdr: self.sdr.net_infer(&mut ecc.ecc,learn.unwrap_or(false)) }
//     }
//     #[text_signature = "(start,end)"]
//     pub fn subdataset(&self, start: usize, end: Option<usize>) -> Self {
//         let end = end.unwrap_or(self.sdr.len());
//         Self { sdr: self.sdr.subdataset(start..end) }
//     }
//
//     // #[text_signature = "(ecc_dense,target)"]
//     // fn batch_infer_conv_weights(&self, ecc: &ConvWeights, target: &CpuEccPopulation) -> CpuSdrDataset {
//     //     Self { sdr: self.sdr.batch_infer_conv_weights(&ecc.ecc, target.ecc.clone()) }
//     // }
//     // #[text_signature = "(ecc_dense)"]
//     // fn batch_infer_and_measure_s_expectation(&self, ecc: &EccLayer) -> (CpuSdrDataset, f32, u32) {
//     //     let (sdr, s_exp, missed) = self.sdr.batch_infer_and_measure_s_expectation(&ecc.ecc);
//     //     (Self { sdr }, s_exp, missed)
//     // }
//     // #[text_signature = "(ecc_dense,target)"]
//     // fn batch_infer_conv_weights_and_measure_s_expectation(&self, ecc: &ConvWeights, target: &CpuEccPopulation) -> (CpuSdrDataset, f32, u32) {
//     //     let (sdr, s_exp, missed) = self.sdr.batch_infer_conv_weights_and_measure_s_expectation(&ecc.ecc, target.ecc.clone());
//     //     (Self { sdr }, s_exp, missed)
//     // }
//     #[text_signature = "(labels,number_of_classes,invariant_to_column)"]
//     fn fit_naive_bayes(&self, labels: &PyAny, number_of_classes: usize, invariant_to_column: Option<bool>) -> PyResult<Occurrences> {
//         let invariant_to_column = invariant_to_column.unwrap_or(false);
//         let array = unsafe {
//             if npyffi::PyArray_Check(labels.as_ptr()) == 0 {
//                 return Err(PyDowncastError::new(labels, "PyArray<T, D>").into());
//             }
//             &*(labels as *const PyAny as *const PyArrayDyn<u8>)
//         };
//         if !array.is_c_contiguous() {
//             return Err(PyValueError::new_err("Numpy array is not C contiguous"));
//         }
//         let dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
//         fn f<T: Element>(sdr: &htm::CpuSdrDataset, labels: &PyAny, number_of_classes: usize, invariant_to_column: bool, f: impl Fn(&T) -> usize) -> PyResult<htm::Occurrences> {
//             let labels = unsafe { &*(labels as *const PyAny as *const PyArrayDyn<T>) };
//             let labels = unsafe { labels.as_slice()? };
//             Ok(sdr.fit_naive_bayes(labels, number_of_classes, invariant_to_column, f))
//         }
//         match dtype {
//             u8::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &u8| *f as usize),
//             u32::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &u32| *f as usize),
//             u64::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &u64| *f as usize),
//             i8::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &i8| *f as usize),
//             i32::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &i32| *f as usize),
//             i64::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &i64| *f as usize),
//             usize::DATA_TYPE => f(&self.sdr, labels, number_of_classes, invariant_to_column, |f: &usize| *f),
//             d => Err(PyValueError::new_err(format!("Unexpected dtype {:?} of numpy array ", d)))
//         }.map(|c| Occurrences { c })
//     }
//
// }
//
// #[pyproto]
// impl PyIterProtocol for CpuSdrDataset{
//     fn __iter__(slf: PyRef<Self>) -> PyResult<Py<CpuSdrDatasetIter>> {
//         let iter = CpuSdrDatasetIter {
//             inner: slf.sdr.to_vec().into_iter(),
//         };
//         Py::new(slf.py(), iter)
//     }
// }
// #[pyproto]
// impl PyIterProtocol for CpuSdrDatasetIter {
//     fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
//         slf
//     }
//
//     fn __next__(mut slf: PyRefMut<Self>) -> Option<CpuSDR> {
//         slf.inner.next().map(|sdr|CpuSDR{sdr})
//     }
// }
// #[pyclass]
// pub struct CpuSdrDatasetIter {
//     inner: std::vec::IntoIter<htm::CpuSDR>,
// }
//
//
//
// impl_save_load!(CpuSdrDataset,sdr);
// impl_save_load!(SubregionIndices,sdr);
// impl_save_load!(Occurrences,c);
//
// #[pyproto]
// impl PySequenceProtocol for CpuSdrDataset {
//     fn __len__(&self) -> usize {
//         self.sdr.len()
//     }
//     fn __getitem__(&self, idx: isize) -> CpuSDR {
//         assert!(idx >= 0);
//         CpuSDR { sdr: self.sdr[idx as usize].clone() }
//     }
//
//     fn __setitem__(&mut self, idx: isize, value: PyRef<CpuSDR>) {
//         assert!(idx >= 0);
//         self.sdr[idx as usize] = value.sdr.clone();
//     }
// }
//
//
// #[pyproto]
// impl PySequenceProtocol for SubregionIndices {
//     fn __len__(&self) -> usize {
//         self.sdr.len()
//     }
//     fn __getitem__(&self, idx: isize) -> Vec<u32> {
//         assert!(idx >= 0);
//         self.sdr[idx as usize].to_vec()
//     }
//
//     fn __setitem__(&mut self, idx: isize, value: [Idx; 3]) {
//         assert!(idx >= 0);
//         self.sdr[idx as usize] = value;
//     }
// }