import transform
import streamlit as st
import numpy as np
import tempfile
import os
from pydicom import dcmread
from base64 import b64encode


MASK_SIZE = 11


class MetaData:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.value = None
        self.input = None


META_DATA = [
    MetaData('PatientName', 'Patient\'s name'),
    MetaData('PatientID', 'Patient\'s ID'),
    MetaData('PatientSex', 'Patient\'s sex'),
    MetaData('PatientBirthDate', 'Patient\'s birth date'),
    MetaData('StudyDate', 'Study date'),
    MetaData('AdditionalPatientHistory', 'Comment')
]


REQUIRED_META_DATA = set(data.name for data in META_DATA)


class Result:
    def __init__(self, animate, input_image, meta_data, sinogram, filtered_sinogram, output_image, mse):
        self.animate = animate
        self.input_image = input_image
        self.meta_data = meta_data
        self.sinogram = sinogram
        self.filtered_sinogram = filtered_sinogram
        self.output_image = output_image
        self.mse = mse


@st.cache(show_spinner=True)
def apply_transformation(file, emitter_step, detector_count, detector_span, apply_filter, animate):
    input_image, meta_data = transform.load_img(file)
    meta_data = {name: meta_data.get(name, "-") for name in REQUIRED_META_DATA}
    img, offset = transform.resize_to_square(input_image)
    r = transform.calc_radius(img)
    emitter_step_rad = np.deg2rad(emitter_step)
    detector_span_rad = np.deg2rad(detector_span)
    sinogram = transform.img_to_sinogram(img, emitter_step_rad, r, detector_count,
                                         detector_span_rad)
    if apply_filter:
        filtered_sinogram = transform.convolve(sinogram, MASK_SIZE)
        used_sinogram = filtered_sinogram
    else:
        filtered_sinogram = None
        used_sinogram = sinogram
    output_image = transform.sinogram_to_img(used_sinogram, animate,
                                             emitter_step_rad, r, detector_count,
                                             detector_span_rad, offset)
    mse = transform.mean_square_error(input_image, output_image)
    return Result(animate, input_image, meta_data, sinogram,
                  transform.normalize_img(filtered_sinogram),
                  output_image, mse)


def show_input_img(img, meta_data):
    expander = st.beta_expander("Input")
    col1, col2 = expander.beta_columns(2)
    col1.image(img, "Input image", width=300)
    for data in META_DATA:
        col2.markdown(f"**{data.label}**: {meta_data[data.name]}")


def show_sinogram(sinogram, filtered_sinogram, animate):
    expander = st.beta_expander("Sinogram")
    col1, col2 = expander.beta_columns(2)
    img_elem = col1.empty()
    if animate:
        slider = col1.slider("Step", 0, sinogram.shape[0], key="sinogram-slider",
                             value=sinogram.shape[0], step=1)
        sin_copy = np.zeros(sinogram.shape)
        sin_copy[:slider, :] = sinogram[:slider, :]
    else:
        sin_copy = sinogram
    img_elem.image(sin_copy, "Sinogram", width=300)

    if filtered_sinogram is not None:
        col2.image(filtered_sinogram, "Filtered sinogram", width=300)


@st.cache
def save_file(image):
    filename = tempfile.NamedTemporaryFile(suffix='.dcm').name
    ds = dcmread("Tomograf_DICOM.dcm")
    ds.Rows = image.shape[0]
    ds.Columns = image.shape[1]
    ds.PixelData = np.asarray(image * 255, dtype=np.uint16).tobytes()
    ds.PatientName = META_DATA[0].input
    ds.PatientID = META_DATA[1].input
    ds.InstitutionName = 'Politechnika Poznanska'
    ds.Manufacturer = 'Politechnika Poznanska'
    ds.PatientSex = META_DATA[2].input
    ds.PatientBirthDate = META_DATA[3].input
    ds.StudyDate = META_DATA[4].input
    ds.AdditionalPatientHistory = META_DATA[5].input
    ds.save_as(filename)

    with open(filename, "rb") as file:
        data = file.read()
        encoded = b64encode(data).decode()
        link = f'<a href="data:file/dcm;base64,{encoded}" download="output.dcm">Download File</a>'
    os.remove(filename)
    return link


def show_output_image(output_image, animate, meta_data, mse):
    expander = st.beta_expander("Output")
    col1, col2 = expander.beta_columns(2)

    img_elem = col1.empty()
    if animate:
        slider = col1.slider("Step", 1, output_image.shape[0], key="output-slider",
                             value=output_image.shape[0], step=1)
        img = output_image[slider - 1]
        col1.markdown("**Mean squared error**")
        col1.line_chart(mse)
    else:
        img = output_image
        col1.markdown(f"**Mean squared error**: {mse}")
    img_elem.image(img, "Output", width=300)

    for data in META_DATA:
        data.input = col2.text_input(data.label, meta_data[data.name], key=f"text-input-{data.name}")

    if expander.button("Generate DICOM file"):
        link = save_file(output_image)
        expander.markdown(link, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Computed Tomography Simulator")
    st.title("Computed Tomography Simulator")
    params_expander = st.sidebar.beta_expander("Parameters", expanded=True)
    emitter_step = params_expander.number_input("Emitter step (deg.)", min_value=0.25,
                                                max_value=179.75, step=0.25, value=1.0)
    detector_count = params_expander.number_input("Detector count", min_value=1, max_value=720,
                                                  step=1, value=90)
    detector_span = params_expander.number_input("Detector span (deg.)", min_value=0.25,
                                                 max_value=270.0, step=0.25, value=180.0)
    apply_filter = params_expander.checkbox("Filter")
    animate = params_expander.checkbox("Animate")

    file = st.sidebar.file_uploader("Choose file", type=('png', 'jpg', 'dcm'))
    if file is not None:
        result = apply_transformation(file, emitter_step, detector_count,
                                      detector_span, apply_filter, animate)
        show_input_img(result.input_image, result.meta_data)
        show_sinogram(result.sinogram, result.filtered_sinogram, result.animate)
        show_output_image(result.output_image, result.animate, result.meta_data, result.mse)
    else:
        st.text("No file")


if __name__ == '__main__':
    main()
