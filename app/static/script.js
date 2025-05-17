document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const preview = document.getElementById('preview');
    const imagePreview = document.getElementById('imagePreview');
    const removeImage = document.getElementById('removeImage');
    const result = document.getElementById('result');
    const loading = document.getElementById('loading');
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');
    const probabilities = document.getElementById('probabilities');

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    preview.classList.remove('hidden');
                    uploadArea.classList.add('hidden');
                    analyzeImage(file);
                };
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    removeImage.addEventListener('click', () => {
        fileInput.value = '';
        preview.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        result.classList.add('hidden');
    });

    async function analyzeImage(file) {
        loading.classList.remove('hidden');
        result.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to analyze image');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing image. Please try again.');
        } finally {
            loading.classList.add('hidden');
        }
    }

    function displayResults(data) {
        prediction.textContent = data.prediction;
        confidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;

        // Clear previous probabilities
        probabilities.innerHTML = '';

        // Add probability rows for each class
        Object.entries(data.all_probabilities)
            .sort(([, a], [, b]) => b - a)
            .forEach(([className, prob]) => {
                const probPercent = (prob * 100).toFixed(2);
                const div = document.createElement('div');
                div.className = 'result-row mb-2';
                if (className === data.prediction) {
                    div.classList.add('highlight');
                }
                div.innerHTML = `
                    <span class="text-sm font-medium text-gray-700">${className}</span>
                    <span class="text-sm text-gray-500">${probPercent}%</span>
                `;
                probabilities.appendChild(div);
            });

        result.classList.remove('hidden');
    }
}); 