import streamlit as st
from PyPDF2 import PdfMerger

# Streamlit Page Config
st.set_page_config(page_title="📎 PDF Merger Utility", layout="centered")
st.title("📎 Merge Multiple PDFs into One")

# File Upload
uploaded_files = st.file_uploader("Upload PDF files to merge", type="pdf", accept_multiple_files=True)

# Merge PDFs
if uploaded_files and st.button("🔗 Merge PDFs"):
    merger = PdfMerger()

    for uploaded_file in uploaded_files:
        merger.append(uploaded_file)

    merged_filename = "merged_result.pdf"
    merger.write(merged_filename)
    merger.close()

    with open(merged_filename, "rb") as f:
        st.success("✅ PDFs successfully merged!")
        st.download_button(
            label="📥 Download Merged PDF",
            data=f,
            file_name=merged_filename,
            mime="application/pdf"
        )
