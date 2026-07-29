/* ============================================================
   Leaf Recognition AI — authentication page logic
   Talks only to the local backend:
     POST /api/auth/signup      -> { token, user }
     POST /api/auth/login       -> { token, user }
     POST /api/auth/login-face  -> { token, user, similarity }
     GET  /api/auth/me          -> { user }
   On success the bearer token is saved to localStorage and the browser
   is redirected to the main app.
   ============================================================ */
"use strict";

/* ---------------- shared session keys (kept in sync with app.js) ------- */
const TOKEN_KEY = "leaf_auth_token";
const USER_KEY = "leaf_auth_user";
const APP_URL = "index.html";

/* ---------------- tiny helpers ---------------- */
const $ = (id) => document.getElementById(id);
const show = (el) => { if (el) el.hidden = false; };
const hide = (el) => { if (el) el.hidden = true; };

function saveSession(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user || {}));
}

async function apiJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status}).`);
  return data;
}

/* ---------------- element refs ---------------- */
const el = {
  title: $("auth-title"),
  sub: $("auth-sub"),
  tabLogin: $("tab-login"),
  tabSignup: $("tab-signup"),
  error: $("auth-error"),
  success: $("auth-success"),
  // login
  loginForm: $("login-form"),
  loginEmail: $("login-email"),
  loginPassword: $("login-password"),
  loginSubmit: $("login-submit"),
  faceLoginBtn: $("face-login-btn"),
  // signup
  signupForm: $("signup-form"),
  signupName: $("signup-name"),
  signupEmail: $("signup-email"),
  signupPassword: $("signup-password"),
  signupConfirm: $("signup-confirm"),
  signupSubmit: $("signup-submit"),
  enrollToggle: $("enroll-toggle"),
  enrollHint: $("enroll-hint"),
  enrollStatus: $("enroll-status"),
  // face modal
  faceModal: $("face-modal"),
  faceModalTitle: $("face-modal-title"),
  faceVideo: $("face-video"),
  faceModalError: $("face-modal-error"),
  faceCaptureBtn: $("face-capture-btn"),
};

/* ---------------- state ---------------- */
const state = {
  camStream: null,
  purpose: "login",     // "login" | "enroll"
  signupFace: null,     // captured data URL for signup enrollment
};

/* ============================================================
   Feedback banners
   ============================================================ */
function fail(msg) {
  hide(el.success);
  el.error.textContent = msg;
  show(el.error);
}
function ok(msg) {
  hide(el.error);
  el.success.textContent = msg;
  show(el.success);
}
function clearBanners() { hide(el.error); hide(el.success); }

/* ============================================================
   Tab switching
   ============================================================ */
function setMode(mode) {
  clearBanners();
  const isLogin = mode === "login";
  el.tabLogin.classList.toggle("is-active", isLogin);
  el.tabSignup.classList.toggle("is-active", !isLogin);
  el.tabLogin.setAttribute("aria-selected", String(isLogin));
  el.tabSignup.setAttribute("aria-selected", String(!isLogin));
  el.loginForm.hidden = !isLogin;
  el.signupForm.hidden = isLogin;
  el.title.textContent = isLogin ? "Welcome back" : "Create your account";
  el.sub.textContent = isLogin
    ? "Sign in to identify plants and diagnose leaf health."
    : "It only takes a moment. Everything stays on this machine.";
}

/* ============================================================
   Email + password login
   ============================================================ */
async function handleLogin(e) {
  e.preventDefault();
  clearBanners();
  const email = el.loginEmail.value.trim();
  const password = el.loginPassword.value;
  if (!email || !password) return fail("Enter your email and password.");

  el.loginSubmit.disabled = true;
  el.loginSubmit.textContent = "Signing in…";
  try {
    const data = await apiJSON("/api/auth/login", { email, password });
    saveSession(data.token, data.user);
    window.location.href = APP_URL;
  } catch (err) {
    fail(err.message);
  } finally {
    el.loginSubmit.disabled = false;
    el.loginSubmit.textContent = "Sign in";
  }
}

/* ============================================================
   Sign up (+ optional face enrollment)
   ============================================================ */
async function handleSignup(e) {
  e.preventDefault();
  clearBanners();
  const name = el.signupName.value.trim();
  const email = el.signupEmail.value.trim();
  const password = el.signupPassword.value;
  const confirm = el.signupConfirm.value;

  if (!name) return fail("Enter your name.");
  if (!email) return fail("Enter your email.");
  if (password.length < 8) return fail("Password must be at least 8 characters.");
  if (password !== confirm) return fail("Passwords do not match.");
  if (el.enrollToggle.checked && !state.signupFace) {
    return fail("Capture your face to finish face setup, or turn it off.");
  }

  const body = { name, email, password };
  if (el.enrollToggle.checked && state.signupFace) body.face_image = state.signupFace;

  el.signupSubmit.disabled = true;
  el.signupSubmit.textContent = "Creating account…";
  try {
    const data = await apiJSON("/api/auth/signup", body);
    saveSession(data.token, data.user);
    window.location.href = APP_URL;
  } catch (err) {
    fail(err.message);
  } finally {
    el.signupSubmit.disabled = false;
    el.signupSubmit.textContent = "Create account";
  }
}

/* ============================================================
   Face capture modal (used for both login and enrollment)
   ============================================================ */
async function openFaceModal(purpose) {
  state.purpose = purpose;
  clearBanners();
  hide(el.faceModalError);
  el.faceModalTitle.textContent =
    purpose === "login" ? "Sign in with your face" : "Set up face login";
  el.faceCaptureBtn.textContent = purpose === "login" ? "Capture & sign in" : "Capture";
  show(el.faceModal);
  try {
    state.camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    el.faceVideo.srcObject = state.camStream;
  } catch (err) {
    el.faceModalError.textContent =
      `Camera unavailable (${err.name}). Use email and password instead.`;
    show(el.faceModalError);
  }
}

function closeFaceModal() {
  if (state.camStream) {
    state.camStream.getTracks().forEach((t) => t.stop());
    state.camStream = null;
  }
  hide(el.faceModal);
}

function grabFrame() {
  const v = el.faceVideo;
  if (!v.videoWidth) return null;
  const canvas = document.createElement("canvas");
  canvas.width = v.videoWidth;
  canvas.height = v.videoHeight;
  canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.9);
}

async function handleFaceCapture() {
  const dataURL = grabFrame();
  if (!dataURL) {
    el.faceModalError.textContent = "Camera not ready yet — try again.";
    show(el.faceModalError);
    return;
  }

  if (state.purpose === "enroll") {
    state.signupFace = dataURL;
    closeFaceModal();
    el.enrollStatus.textContent = "Face captured ✓ — tap the toggle to retake.";
    show(el.enrollStatus);
    return;
  }

  // purpose === "login"
  el.faceCaptureBtn.disabled = true;
  el.faceCaptureBtn.textContent = "Matching…";
  try {
    const data = await apiJSON("/api/auth/login-face", { face_image: dataURL });
    saveSession(data.token, data.user);
    closeFaceModal();
    window.location.href = APP_URL;
  } catch (err) {
    el.faceModalError.textContent = err.message;
    show(el.faceModalError);
  } finally {
    el.faceCaptureBtn.disabled = false;
    el.faceCaptureBtn.textContent = "Capture & sign in";
  }
}

/* ============================================================
   Already-signed-in guard: skip the login page if the token is valid
   ============================================================ */
async function redirectIfAuthenticated() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) return;
  try {
    const res = await fetch("/api/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) window.location.href = APP_URL;
  } catch (_) {
    /* offline / invalid — stay on the login page */
  }
}

/* ============================================================
   Event wiring
   ============================================================ */
el.tabLogin.addEventListener("click", () => setMode("login"));
el.tabSignup.addEventListener("click", () => setMode("signup"));

el.loginForm.addEventListener("submit", handleLogin);
el.signupForm.addEventListener("submit", handleSignup);
el.faceLoginBtn.addEventListener("click", () => openFaceModal("login"));
el.faceCaptureBtn.addEventListener("click", handleFaceCapture);

el.enrollToggle.addEventListener("change", () => {
  if (el.enrollToggle.checked) {
    show(el.enrollHint);
    openFaceModal("enroll");
  } else {
    hide(el.enrollHint);
    hide(el.enrollStatus);
    state.signupFace = null;
  }
});

document.querySelectorAll("[data-close]").forEach((node) => {
  node.addEventListener("click", () => closeFaceModal());
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeFaceModal();
});

redirectIfAuthenticated();
