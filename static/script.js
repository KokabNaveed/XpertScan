const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput); // Append file input to the body

    const dropArea = document.getElementById('drop-area');
    const preview = document.getElementById('preview');
    const submitButton = document.getElementById('submit-button');

    // Open file dialog when drop area is clicked
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', handleFiles);
    
    // Handle drag and drop
    dropArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropArea.classList.add('hover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('hover');
    });

    dropArea.addEventListener('drop', (event) => {
        event.preventDefault();
        dropArea.classList.remove('hover');
        const files = event.dataTransfer.files;
        if (files.length) {
            fileInput.files = files; // Set the dropped files to the input
            handleFiles(); // Call the function to handle the files
        }
    });

    // Function to handle file selection and preview
    function handleFiles() {
        const files = fileInput.files;
        if (files.length) {
            const file = files[0];
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = 'block'; // Show the preview
                submitButton.disabled = false; // Enable the submit button
            };
            reader.readAsDataURL(file);
        }
    }

    // Optionally handle submit button click
    submitButton.addEventListener('click', () => {
        if (!submitButton.disabled) {
            // Handle the submit logic here
            alert('Submitting: ' + fileInput.files[0].name);
        }
    });