/* ============================================================
   Leaf Recognition AI — client logic
   Talks only to the local backend:
     POST /api/analyze   -> { species: {...}, health: {...} }
     GET  /thumbs/<file> -> dataset thumbnails
   Everything happens dynamically; the page never reloads.
   ============================================================ */
"use strict";

/* ---------------- constants ---------------- */
const API_ANALYZE = "/api/analyze";
const THUMB_BASE = "/thumbs/";
const MAX_BYTES = 15 * 1024 * 1024;
const ACCEPTED = ["image/png", "image/jpeg"];
const RING_CIRCUMFERENCE = 2 * Math.PI * 52; // r = 52 in the SVG

const LOADING_STEPS = [
  { msg: "Extracting features…", progress: 20 },
  { msg: "Identifying species…", progress: 45 },
  { msg: "Comparing with dataset…", progress: 70 },
  { msg: "Preparing results…", progress: 90 },
];

/* ---------------- tiny DOM helpers ---------------- */
const $ = (id) => document.getElementById(id);
const show = (elm) => { elm.hidden = false; };
const hide = (elm) => { elm.hidden = true; };
const pct = (x) => Math.round((Number(x) || 0) * 100);

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

/* map backend health status -> UI badge semantics */
function statusMeta(status) {
  switch (status) {
    case "HEALTHY":       return { key: "healthy",  label: "Healthy" };
    case "MINOR_ISSUES":  return { key: "minor",    label: "Minor Issues" };
    case "UNHEALTHY":     return { key: "diseased", label: "Diseased" };
    default:              return { key: "unknown",  label: "Unknown" };
  }
}

/* ---------------- element refs ---------------- */
const el = {
  panels: {
    upload: $("upload-panel"),
    loading: $("loading-panel"),
    results: $("results-panel"),
    similar: $("similar-panel"),
  },
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  browseBtn: $("browse-btn"),
  cameraBtn: $("camera-btn"),
  heroUpload: $("hero-upload"),
  uploadError: $("upload-error"),
  // preview
  previewCard: $("preview-card"),
  previewImg: $("preview-img"),
  previewName: $("preview-name"),
  previewSize: $("preview-size"),
  analyzeBtn: $("analyze-btn"),
  replaceBtn: $("replace-btn"),
  removeBtn: $("remove-btn"),
  // loading
  loadingMsg: $("loading-msg"),
  progressBar: $("progress-bar"),
  // results
  resultImg: $("result-img"),
  healthBadge: $("health-badge"),
  severityBadge: $("severity-badge"),
  plantName: $("plant-name"),
  plantDesc: $("plant-desc"),
  ringFill: $("ring-fill"),
  confValue: $("conf-value"),
  altSpecies: $("alt-species"),
  diagnosisCard: $("diagnosis-card"),
  diseaseName: $("disease-name"),
  dSeverity: $("d-severity"),
  dSymptoms: $("d-symptoms"),
  dTreatment: $("d-treatment"),
  dPrevention: $("d-prevention"),
  analysisSource: $("analysis-source"),
  analysisDisclaimer: $("analysis-disclaimer"),
  againBtn: $("again-btn"),
  // similar
  similarGrid: $("similar-grid"),
  // camera modal
  cameraModal: $("camera-modal"),
  cameraVideo: $("camera-video"),
  cameraError: $("camera-error"),
  captureBtn: $("capture-btn"),
  // image / compare modal
  imageModal: $("image-modal"),
  modalSingle: $("modal-single"),
  modalCompare: $("modal-compare"),
  modalImg: $("modal-img"),
  modalName: $("modal-name"),
  modalSimilarity: $("modal-similarity"),
  compareBtn: $("compare-btn"),
  compareBack: $("compare-back"),
  cmpYourImg: $("cmp-your-img"),
  cmpRefImg: $("cmp-ref-img"),
  cmpRing: $("cmp-ring"),
  cmpScore: $("cmp-score"),
  compareTable: $("compare-table"),
  connBadge: $("conn-badge"),
};

/* ---------------- app state ---------------- */
const state = {
  file: null,          // currently selected Blob/File
  previewURL: null,    // object URL for the uploaded image
  result: null,        // last /api/analyze response
  activeItem: null,    // similar item open in the modal
  camStream: null,
  loadingTimer: null,
};

/* ============================================================
   Image selection + preview
   ============================================================ */

/** Validate and adopt an image (from browse, drag-drop, or camera). */
function uploadImage(fileOrBlob) {
  const type = fileOrBlob.type || "image/jpeg";
  if (!ACCEPTED.includes(type)) {
    return fail("Unsupported format. Please use PNG, JPG or JPEG.");
  }
  if (fileOrBlob.size > MAX_BYTES) {
    return fail("Image is too large (max 15 MB).");
  }
  clearError();
  state.file = fileOrBlob;
  previewImage(fileOrBlob);
}

/** Render the selected image preview with name + size. */
function previewImage(fileOrBlob) {
  if (state.previewURL) URL.revokeObjectURL(state.previewURL);
  state.previewURL = URL.createObjectURL(fileOrBlob);

  el.previewImg.src = state.previewURL;
  el.previewName.textContent = fileOrBlob.name || "captured-leaf.jpg";
  el.previewSize.textContent = formatBytes(fileOrBlob.size);
  show(el.previewCard);
  el.previewCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/** Clear the current selection and return to the empty dropzone. */
function removeImage() {
  state.file = null;
  if (state.previewURL) { URL.revokeObjectURL(state.previewURL); state.previewURL = null; }
  el.fileInput.value = "";
  hide(el.previewCard);
  clearError();
}

/* ============================================================
   Prediction
   ============================================================ */

async function sendPrediction() {
  if (!state.file) return;

  // keep a stable image URL for the results + compare views
  const resultURL = state.previewURL;
  el.resultImg.src = resultURL;

  showLoading();
  hide(el.panels.results);
  hide(el.panels.similar);

  const form = new FormData();
  form.append("image", state.file, state.file.name || "leaf.jpg");

  try {
    const res = await fetch(API_ANALYZE, { method: "POST", body: form });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Analysis failed (${res.status}).`);
    }
    const data = await res.json();
    state.result = data;
    await hideLoading();
    displayResults(data);
  } catch (err) {
    await hideLoading();
    fail(err.message || "Something went wrong. Please try again.");
    hide(el.panels.results);
    hide(el.panels.similar);
  }
}

/* ============================================================
   Loading state (spinner + staged progress + rotating messages)
   ============================================================ */

function showLoading() {
  show(el.panels.loading);
  el.panels.loading.scrollIntoView({ behavior: "smooth", block: "center" });
  el.progressBar.style.width = "6%";
  el.loadingMsg.textContent = LOADING_STEPS[0].msg;

  let i = 0;
  clearInterval(state.loadingTimer);
  state.loadingTimer = setInterval(() => {
    i = Math.min(i + 1, LOADING_STEPS.length - 1);
    const step = LOADING_STEPS[i];
    el.loadingMsg.style.opacity = "0";
    setTimeout(() => {
      el.loadingMsg.textContent = step.msg;
      el.loadingMsg.style.opacity = "1";
    }, 180);
    el.progressBar.style.width = step.progress + "%";
    if (i === LOADING_STEPS.length - 1) clearInterval(state.loadingTimer);
  }, 900);
}

function hideLoading() {
  clearInterval(state.loadingTimer);
  el.progressBar.style.width = "100%";
  return new Promise((resolve) => {
    setTimeout(() => { hide(el.panels.loading); resolve(); }, 350);
  });
}

/* ============================================================
   Render results
   ============================================================ */

function setRing(circleEl, labelEl, percent) {
  const p = Math.max(0, Math.min(100, percent));
  // force a reflow so the transition animates from 0 each time
  circleEl.style.strokeDashoffset = RING_CIRCUMFERENCE;
  void circleEl.getBoundingClientRect();
  circleEl.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - p / 100);
  if (labelEl) labelEl.textContent = p + "%";
}

function displayResults(data) {
  const sp = data.species || {};
  const h = data.health || {};
  const care = h.care || {};
  const meta = statusMeta(h.status);

  // --- badges ---
  el.healthBadge.dataset.status = meta.key;
  el.healthBadge.textContent = meta.label;
  if (care.severity && care.severity !== "None" && care.severity !== "Unknown") {
    el.severityBadge.textContent = "Severity: " + care.severity;
    show(el.severityBadge);
  } else {
    hide(el.severityBadge);
  }

  // --- species ---
  el.plantName.textContent = sp.species || "Unknown";
  // Description is the local vision model's AI-generated observation of the
  // captured leaf. When the model isn't running it comes back empty.
  el.plantDesc.textContent = sp.description
    ? sp.description
    : "AI leaf description unavailable — start the local vision model to generate one.";

  // --- confidence ring ---
  const confPct = pct(sp.confidence);
  el.confValue.textContent = "0%";
  requestAnimationFrame(() => setRing(el.ringFill, el.confValue, confPct));

  // --- alternative species ---
  el.altSpecies.innerHTML = "";
  (sp.top_species || []).slice(1).forEach((a) => {
    const li = document.createElement("li");
    li.textContent = `${a.species} · ${pct(a.score)}%`;
    el.altSpecies.appendChild(li);
  });

  // --- diagnosis ---
  el.diagnosisCard.dataset.status = meta.key;
  el.diseaseName.textContent = care.disease || meta.label;
  el.dSeverity.textContent = care.severity || "—";
  el.dSymptoms.textContent = care.symptoms || "—";
  el.dTreatment.textContent = care.treatment || "—";
  el.dPrevention.textContent = care.prevention || "—";

  el.analysisSource.textContent = h.vlm_used
    ? "Assessed by a local AI vision model."
    : "Local AI vision model not running — status shown is an automated visual estimate.";
  el.analysisDisclaimer.textContent = h.disclaimer || "";

  // --- reveal ---
  show(el.panels.results);
  el.panels.results.scrollIntoView({ behavior: "smooth", block: "start" });

  renderSimilarImages(sp.similar || []);
}

/* ============================================================
   Similar leaves
   ============================================================ */

function renderSimilarImages(items) {
  el.similarGrid.innerHTML = "";
  if (!items.length) { hide(el.panels.similar); return; }

  items.forEach((item, idx) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "leaf-card";
    card.style.animationDelay = idx * 70 + "ms"; // staggered entrance
    card.setAttribute("aria-label", `${item.species}, ${pct(item.similarity)}% similar. Open preview.`);
    card.innerHTML =
      `<div class="leaf-card__img"><img loading="lazy" src="${THUMB_BASE}${encodeURIComponent(item.thumb)}" alt="${item.species}"></div>` +
      `<div class="leaf-card__body">` +
        `<p class="leaf-card__name">${item.species}</p>` +
        `<div class="leaf-card__row">` +
          `<span class="tag">Reference</span>` +
          `<span class="leaf-card__sim">${pct(item.similarity)}% match</span>` +
        `</div>` +
      `</div>`;
    card.addEventListener("click", () => openImageModal(item));
    el.similarGrid.appendChild(card);
  });

  show(el.panels.similar);
}

/* ============================================================
   Image modal + compare
   ============================================================ */

function openImageModal(item) {
  state.activeItem = item;
  el.modalImg.src = THUMB_BASE + encodeURIComponent(item.thumb);
  el.modalImg.alt = item.species;
  el.modalName.textContent = item.species;
  el.modalSimilarity.textContent = `${pct(item.similarity)}% similar to your leaf`;
  show(el.modalSingle);
  hide(el.modalCompare);
  show(el.imageModal);
}

function compareImages(item) {
  if (!state.result) return;
  const sp = state.result.species || {};
  const meta = statusMeta((state.result.health || {}).status);

  el.cmpYourImg.src = state.previewURL || el.resultImg.src;
  el.cmpRefImg.src = THUMB_BASE + encodeURIComponent(item.thumb);

  const simPct = pct(item.similarity);
  el.cmpScore.textContent = "0%";
  requestAnimationFrame(() => setRing(el.cmpRing, el.cmpScore, simPct));

  const rows = [
    ["", "Your leaf", "Reference"],
    ["Plant", sp.species || "Unknown", item.species],
    ["Status", meta.label, "Healthy sample"],
    ["Confidence", pct(sp.confidence) + "%", simPct + "% match"],
  ];
  el.compareTable.innerHTML = rows
    .map((r, i) =>
      `<div class="row${i === 0 ? " head" : ""}">` +
      r.map((c) => `<span>${c}</span>`).join("") +
      `</div>`
    )
    .join("");

  hide(el.modalSingle);
  show(el.modalCompare);
}

/* ============================================================
   Camera capture (secondary input path)
   ============================================================ */

async function openCamera() {
  clearCamError();
  show(el.cameraModal);
  try {
    state.camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    el.cameraVideo.srcObject = state.camStream;
  } catch (err) {
    el.cameraError.textContent =
      `Camera unavailable (${err.name}). You can still upload an image.`;
    show(el.cameraError);
  }
}

function closeCamera() {
  if (state.camStream) {
    state.camStream.getTracks().forEach((t) => t.stop());
    state.camStream = null;
  }
  hide(el.cameraModal);
}

function captureFromCamera() {
  const v = el.cameraVideo;
  if (!v.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = v.videoWidth;
  canvas.height = v.videoHeight;
  canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (blob) {
      blob.name = "captured-leaf.jpg";
      closeCamera();
      uploadImage(blob);
    }
  }, "image/jpeg", 0.9);
}

/* ============================================================
   Errors / connectivity
   ============================================================ */

function fail(message) {
  el.uploadError.textContent = message;
  show(el.uploadError);
}
function clearError() { hide(el.uploadError); }
function clearCamError() { hide(el.cameraError); }

function updateConnBadge() {
  el.connBadge.textContent = navigator.onLine ? "Local" : "Offline";
}

/* ============================================================
   Reset
   ============================================================ */

function scanAnother() {
  removeImage();
  hide(el.panels.results);
  hide(el.panels.similar);
  el.panels.upload.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ============================================================
   Event wiring
   ============================================================ */

function openFilePicker() { el.fileInput.click(); }

el.heroUpload.addEventListener("click", () => {
  el.panels.upload.scrollIntoView({ behavior: "smooth", block: "start" });
  openFilePicker();
});
el.browseBtn.addEventListener("click", (e) => { e.stopPropagation(); openFilePicker(); });
el.dropzone.addEventListener("click", (e) => {
  if (e.target.closest("button")) return; // ignore inner buttons
  openFilePicker();
});
el.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openFilePicker(); }
});

el.fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) uploadImage(file);
});

// drag & drop
["dragenter", "dragover"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    el.dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    if (evt === "dragleave" && el.dropzone.contains(e.relatedTarget)) return;
    el.dropzone.classList.remove("is-dragover");
  })
);
el.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) uploadImage(file);
});

el.analyzeBtn.addEventListener("click", sendPrediction);
el.replaceBtn.addEventListener("click", openFilePicker);
el.removeBtn.addEventListener("click", removeImage);
el.againBtn.addEventListener("click", scanAnother);

// camera
el.cameraBtn.addEventListener("click", (e) => { e.stopPropagation(); openCamera(); });
el.captureBtn.addEventListener("click", captureFromCamera);

// image / compare modal
el.compareBtn.addEventListener("click", () => compareImages(state.activeItem));
el.compareBack.addEventListener("click", () => { show(el.modalSingle); hide(el.modalCompare); });

// generic modal close (backdrop, close buttons, cancel)
document.querySelectorAll("[data-close]").forEach((node) => {
  node.addEventListener("click", () => {
    const which = node.getAttribute("data-close");
    if (which === "camera") closeCamera();
    else hide(el.imageModal);
  });
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeCamera(); hide(el.imageModal); }
});

// connectivity badge
window.addEventListener("online", updateConnBadge);
window.addEventListener("offline", updateConnBadge);
updateConnBadge();

// PWA service worker (offline shell)
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () =>
    navigator.serviceWorker.register("sw.js").catch(() => {})
  );
}
