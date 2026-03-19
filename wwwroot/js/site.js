/* AI Research Paper Analyzer — site.js */

function onDragOver(e) {
    e.preventDefault();
    document.getElementById('dropZone').classList.add('dragover');
}
function onDragLeave(e) {
    document.getElementById('dropZone').classList.remove('dragover');
}
function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropZone').classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        const dt = new DataTransfer();
        dt.items.add(file);
        document.getElementById('File').files = dt.files;
        document.getElementById('fileLabel').innerHTML =
            '<span class="file-selected-label"><i class="bi bi-file-earmark-check me-1"></i>' + file.name + '</span>';
        document.getElementById('submitBtn').disabled = false;
    } else {
        alert('Please drop a valid PDF file.');
    }
}