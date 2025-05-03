document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('fileInput');
  const fileName = document.getElementById('fileName');
  const targetVariable = document.getElementById('targetVariable');
  const preprocessButton = document.getElementById('preprocessButton');
  const actionButtons = document.getElementById('actionButtons');
  const downloadButton = document.getElementById('downloadButton');
  const continueButton = document.getElementById('continueButton');
  const message = document.getElementById('message');

  let processedFile = null;
  let originalFile = null;
  let originalTarget = '';

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.csv')) {
      fileName.textContent = file.name;
      fileName.classList.add('file-success');
      preprocessButton.disabled = false;
      showMessage("File loaded successfully.", "success");
    } else {
      fileName.textContent = "No file chosen";
      preprocessButton.disabled = true;
      showMessage("Please select a valid CSV file.", "error");
    }
  });

  preprocessButton.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) return;

    const target = targetVariable.value.trim();
    if (!target) {
      showMessage("Please enter a target variable name.", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = async function (e) {
      const csvText = e.target.result;

      // Clean empty lines
      const cleaned = csvText
        .split("\n")
        .filter(line => line.trim() !== "")
        .join("\n");

      const lines = cleaned.split("\n");
      const headers = lines[0].split(",");

      const processedCsv = lines.join("\n");
      const blob = new Blob([processedCsv], { type: 'text/csv' });

      try {
        const formData = new FormData();
        formData.append('file', new File([blob], file.name)); // override file
        formData.append('target_column', target);

        const response = await fetch('/preprocess', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error("Server returned an error.");
        }

        // Get the processed filename from response headers
        const processedFilename = response.headers.get('X-Processed-Filename');
        processedFile = await response.blob();
        originalFile = new File([blob], file.name);  // Save for later reuse
        originalTarget = target;

        actionButtons.classList.remove('hidden');
        actionButtons.classList.add('visible');
        showMessage(`File preprocessed and received from server! (${processedFilename})`, "success");

      } catch (err) {
        showMessage("Error uploading to server: " + err.message, "error");
      }
    };

    reader.onerror = () => {
      showMessage("Error reading the file.", "error");
    };

    reader.readAsText(file);
  });

  downloadButton.addEventListener('click', () => {
    if (!processedFile) return;

    const url = URL.createObjectURL(processedFile);
    const link = document.createElement("a");
    link.href = url;
    link.download = "processed_file.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  });

  continueButton.addEventListener('click', async () => {
    if (!originalFile || !originalTarget) {
      showMessage("No file to continue with. Please preprocess first.", "error");
      return;
    }
  
    const formData = new FormData();
    formData.append('file', originalFile);
    formData.append('target_column', originalTarget);
    console.log(originalFile, originalTarget);
  
    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });
  
      if (response.redirected) {
        window.location.href = response.url;  // ✅ Follow the redirect manually
      } else {
        const text = await response.text();   // For error messages
        console.error(text);
        showMessage(text, "error");
      }
    } catch (error) {
      console.error("Upload failed:", error);
      showMessage("Upload failed.", "error");
    }
  });
  

  function showMessage(text, type) {
    message.textContent = text;
    message.className = `message ${type}`;
    message.style.display = "block";
    setTimeout(() => {
      message.classList.add('visible');
    }, 10);
  }
});
