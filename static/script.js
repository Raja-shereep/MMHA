// Store results for fusion
let mriResult = null;
let eegResult = null;
let clinicalResult = null;

function updateSummary(type, text, isError=false) {
    const el = document.getElementById(`${type}-summary-status`);
    if(el) {
        el.innerText = text;
        el.className = isError ? "error-text" : "success-text";
    }
}

function showResult(type, html) {
    const content = document.getElementById(`${type}-result`);
    const details = document.getElementById(`${type}-details`);
    if(content && details) {
        content.innerHTML = html;
        details.classList.remove("hidden");
        details.open = true;
    }
}

/* File Input Listeners for UI Feedback */
function setupFileInput(id, filenameId) {
    const input = document.getElementById(id);
    const filenameDisplay = document.getElementById(filenameId);
    
    if (input && filenameDisplay) {
        input.addEventListener("change", function(e) {
            if(e.target.files[0]) {
                // Truncate if too long
                let name = e.target.files[0].name;
                if(name.length > 20) name = name.substring(0, 17) + "...";
                filenameDisplay.innerText = name;
                filenameDisplay.style.color = "var(--text)";
            }
        });
    }
}

setupFileInput("mri-input", "mri-filename");
setupFileInput("eeg-input", "eeg-filename");
// setupFileInput("clinical-input", "clinical-filename"); // Removed - now manual entry


/* MRI Logic */
const mriInput = document.getElementById("mri-input");
const mriPreview = document.getElementById("mri-preview");

if(mriInput) {
    mriInput.addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          mriPreview.innerHTML = `<img src="${e.target.result}" alt="MRI Preview">`;
        };
        reader.readAsDataURL(file);
        
        // Reset previous result
        mriResult = null;
        updateSummary("mri", "Pending");
        document.getElementById("mri-details").classList.add("hidden");
      }
    });
}

/* EEG Logic (Preview) */
const eegInput = document.getElementById("eeg-input");
const eegPreview = document.getElementById("eeg-preview");

if(eegInput) {
    eegInput.addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          eegPreview.innerHTML = `<img src="${e.target.result}" alt="EEG Preview">`;
        };
        reader.readAsDataURL(file);
        
        // Reset previous result
        eegResult = null;
        updateSummary("eeg", "Pending");
        document.getElementById("eeg-details").classList.add("hidden");
      }
    });
}


async function predictMRI() {
  const fileInput = document.getElementById("mri-input");
  
  if (!fileInput.files[0]) {
    alert("Please select an MRI image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  
  updateSummary("mri", "Analyzing...");

  try {
    const response = await fetch("/predict_mri", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (response.ok) {
      mriResult = data;
      updateSummary("mri", "Completed");
      showResult("mri", `
            <p><strong>Class:</strong> ${data.mapped_class} <small>(${data.raw_class})</small></p>
            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
            <p><strong>Risk:</strong> ${data.disease_progression}</p>
      `);
    } else {
        updateSummary("mri", "Error", true);
        showResult("mri", `<p style="color:red">Error: ${data.detail}</p>`);
    }
  } catch (error) {
    updateSummary("mri", "Error", true);
    showResult("mri", `<p style="color:red">Error: ${error.message}</p>`);
  }
}

/* EEG Logic */
async function predictEEG() {
  const fileInput = document.getElementById("eeg-input");

  if (!fileInput.files[0]) {
    alert("Please select an EEG image/graph first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  updateSummary("eeg", "Analyzing...");

  try {
    const response = await fetch("/predict_eeg", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (response.ok) {
        console.log("EEG Data received:", data); // Debugging
        eegResult = data;
        updateSummary("eeg", "Completed");
        
        showResult("eeg", `
            <p><strong>Class:</strong> ${data.mapped_class}</p>
            <p><strong>Prob:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
        `);
    } else {
      updateSummary("eeg", "Error", true);
      showResult("eeg", `<p style="color:red">Error: ${data.detail}</p>`);
    }
  } catch (error) {
    updateSummary("eeg", "Error", true);
    showResult("eeg", `<p style="color:red">Error: ${error.message}</p>`);
  }
}

/* Clinical Logic */
async function predictClinical() {
  updateSummary("clinical", "Analyzing...");

  // Collect form data
  const data = {
      Age: document.getElementById("clin-age").value,
      Gender: document.getElementById("clin-gender").value,
      Ethnicity: document.getElementById("clin-ethnicity").value,
      EducationLevel: document.getElementById("clin-education").value,
      BMI: document.getElementById("clin-bmi").value,
      Smoking: document.getElementById("clin-smoking").value,
      MMSE: document.getElementById("clin-mmse").value,
      ADL: document.getElementById("clin-adl").value,
      
      FamilyHistoryAlzheimers: document.getElementById("clin-family").checked ? 'Yes' : 'No',
      Diabetes: document.getElementById("clin-diabetes").checked ? 'Yes' : 'No',
      Hypertension: document.getElementById("clin-hypertension").checked ? 'Yes' : 'No',
      
      MemoryComplaints: document.getElementById("clin-memory").checked ? 'Yes' : 'No',
      Confusion: document.getElementById("clin-confusion").checked ? 'Yes' : 'No',
      Disorientation: document.getElementById("clin-disorientation").checked ? 'Yes' : 'No',
      Forgetfulness: document.getElementById("clin-forgetfulness").checked ? 'Yes' : 'No'
  };

  try {
    const response = await fetch("/predict_clinical", {
      method: "POST",
      headers: {
          "Content-Type": "application/json"
      },
      body: JSON.stringify(data),
    });
    const result = await response.json();

    if (response.ok) {
        clinicalResult = result;
        updateSummary("clinical", `Risk Assessment: ${result.risk_level}`); // Show actual result in summary
        
        showResult("clinical", `
            <p><strong>Risk Prediction:</strong> ${result.label}</p>
            <p><strong>Probability:</strong> ${(result.probability * 100).toFixed(1)}%</p>
        `);
    } else {
       updateSummary("clinical", "Error", true);
       showResult("clinical", `<p style="color:red">Error: ${result.detail}</p>`);
    }
  } catch (error) {
    updateSummary("clinical", "Error", true);
    showResult("clinical", `<p style="color:red">Error: ${error.message}</p>`);
  }
}

/* Fusion Logic */
async function generateFusion() {
    const resultBox = document.getElementById("fusion-result");
    
    // Auto-analyze if files are present but results are missing
    const mriInput = document.getElementById("mri-input");
    if (mriInput.files[0] && !mriResult) {
        updateSummary("mri", "Auto-Analyzing...");
        await predictMRI();
    }

    const eegInput = document.getElementById("eeg-input");
    if (eegInput.files[0] && !eegResult) {
        updateSummary("eeg", "Auto-Analyzing...");
        await predictEEG();
    }

    // Clinical is manual, so we don't check file input
    if (!clinicalResult) {
        // Optional: Trigger auto-analysis for clinical if data is entered?
        // For now, let's assume user must click analyze. 
        // Or we can just call predictClinical() here if we want to force it.
        // Let's force it to ensure we have the latest form data.
        updateSummary("clinical", "Auto-Analyzing...");
        await predictClinical();
    }

    // Check if at least one result is available
    if (!mriResult && !eegResult && !clinicalResult) {
        alert("Please upload and analyze at least one modality (MRI, EEG, or Clinical) before generating a final report.");
        return;
    }
    
    resultBox.innerHTML = "Generating Final Multimodal Report (Synthesizing available data)...";
    resultBox.classList.remove("hidden");
    
    try {
        const response = await fetch("/predict_fusion", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                mri_result: mriResult, // Partial fusion supported (null is okay)
                eeg_result: eegResult,
                clinical_result: clinicalResult
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultBox.innerHTML = data.report_html;
        } else {
            resultBox.innerHTML = `<p style="color:red">Error: ${data.detail}</p>`;
        }
        
    } catch (error) {
        resultBox.innerHTML = `<p style="color:red">Error: ${error.message}</p>`;
    }
}
