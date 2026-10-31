/**
 * Batch CSV Prediction Controller
 */

let selectedFile = null;
let lastBatchResult = null;

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("batch-file-input");
const btnSelectFile = document.getElementById("btn-select-file");
const fileInfoBar = document.getElementById("file-info-bar");
const fileNameDisplay = document.getElementById("file-name-display");
const fileSizeDisplay = document.getElementById("file-size-display");
const btnRunBatch = document.getElementById("btn-run-batch");
const btnClearBatch = document.getElementById("btn-clear-batch");
const batchForm = document.getElementById("batch-upload-form");
const errorAlert = document.getElementById("batch-error-alert");

const summaryPlaceholder = document.getElementById("batch-summary-placeholder");
const summaryContent = document.getElementById("batch-summary-content");
const statusBadge = document.getElementById("batch-status-badge");
const statTotalRows = document.getElementById("stat-total-rows");
const statMeanUptake = document.getElementById("stat-mean-uptake");
const statModelVer = document.getElementById("stat-model-ver");
const previewTbody = document.getElementById("batch-preview-tbody");
const btnDownloadResults = document.getElementById("btn-download-results");
const btnDownloadSample = document.getElementById("btn-download-sample");

// Drag & drop handlers
if (dropZone) {
    dropZone.addEventListener("click", () => fileInput.click());
    btnSelectFile.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--green)";
            dropZone.style.background = "var(--soft)";
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--line)";
            dropZone.style.background = "#fafbfa";
        });
    });

    dropZone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
}

function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
        showError("Please select a valid .csv file.");
        return;
    }
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    fileInfoBar.style.display = "flex";
    btnRunBatch.disabled = false;
    errorAlert.style.display = "none";
}

function showError(msg) {
    errorAlert.textContent = msg;
    errorAlert.style.display = "block";
}

// Reset form
btnClearBatch.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileInfoBar.style.display = "none";
    btnRunBatch.disabled = true;
    errorAlert.style.display = "none";
    summaryPlaceholder.style.display = "block";
    summaryContent.style.display = "none";
    statusBadge.textContent = "Awaiting File";
    statusBadge.className = "badge-tag";
});

// Run batch upload
batchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    btnRunBatch.disabled = true;
    btnRunBatch.textContent = "Processing Batch...";
    errorAlert.style.display = "none";
    statusBadge.textContent = "Running ML Models...";
    statusBadge.className = "badge-tag";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch("/api/predict/batch", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Batch prediction failed.");
        }

        lastBatchResult = data;
        displayBatchResults(data);
    } catch (err) {
        showError(err.message);
        statusBadge.textContent = "Error";
        statusBadge.className = "badge-tag badge-worst";
    } finally {
        btnRunBatch.disabled = false;
        btnRunBatch.textContent = "Run Batch Predictions →";
    }
});

function displayBatchResults(data) {
    statusBadge.textContent = "Completed";
    statusBadge.className = "badge-tag badge-best";

    summaryPlaceholder.style.display = "none";
    summaryContent.style.display = "block";

    statTotalRows.textContent = data.total_rows;
    statModelVer.textContent = data.model_version || "v1.0-ensemble";

    const rows = data.preview || [];
    let sum = 0;
    rows.forEach(r => sum += (r["Predicted CO2 uptake (mmol/g)"] || 0));
    statMeanUptake.textContent = rows.length > 0 ? (sum / rows.length).toFixed(3) : "0.00";

    previewTbody.innerHTML = "";
    rows.slice(0, 15).forEach((r, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${r["Surface Area (m2/g)"] ?? "-"}</td>
            <td>${r["Temp (°C)"] ?? "-"}</td>
            <td>${r["Pressure (bar)"] ?? "-"}</td>
            <td><b style="color: var(--green);">${r["Predicted CO2 uptake (mmol/g)"]?.toFixed(3) ?? "-"}</b></td>
        `;
        previewTbody.appendChild(tr);
    });
}

// Download enriched results CSV
btnDownloadResults.addEventListener("click", () => {
    if (!lastBatchResult || !lastBatchResult.preview) return;
    const items = lastBatchResult.preview;
    const headers = Object.keys(items[0]);
    const csvRows = [headers.join(",")];

    items.forEach(row => {
        const values = headers.map(h => {
            const val = row[h] ?? "";
            return typeof val === "string" && val.includes(",") ? `"${val}"` : val;
        });
        csvRows.push(values.join(","));
    });

    const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `co2_predictions_batch_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
});

// Download sample CSV
btnDownloadSample.addEventListener("click", () => {
    const sampleHeaders = "Surface Area (m2/g),Total Pore Volume(cm3/g),Micropore Volume (cm3/g),C (%),H (%),N (%),O (%),S (%),Temp (°C),Pressure (bar)\n" +
                          "1420.0,0.74,0.48,78.855,1.5,2.26,15.05,0.0,25.0,1.0\n" +
                          "2150.0,1.12,0.85,82.4,0.9,1.8,14.9,0.0,0.0,1.0\n" +
                          "890.0,0.45,0.32,71.2,2.1,3.4,22.1,1.2,25.0,0.15\n" +
                          "1680.0,0.88,0.61,85.1,1.1,0.9,12.9,0.0,50.0,2.0";
    const blob = new Blob([sampleHeaders], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "co2_batch_input_sample.csv";
    a.click();
    URL.revokeObjectURL(url);
});
