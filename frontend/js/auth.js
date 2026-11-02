const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
if (loginForm) loginForm.addEventListener('submit', async (event) => {
  event.preventDefault(); const status = document.getElementById('login-status');
  const body = Object.fromEntries(new FormData(loginForm));
  try { const response = await fetch('/api/auth/login', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const data = await response.json(); if(!response.ok) throw new Error(data.detail || 'Login failed'); localStorage.setItem('co2_access_token', data.access_token); localStorage.setItem('co2_user', JSON.stringify(data.user)); status.textContent = 'Signed in successfully.'; const requestedNext = new URLSearchParams(window.location.search).get('next'); const safeNext = requestedNext && !requestedNext.startsWith('/') && !requestedNext.includes('://') ? requestedNext : null; window.location.href = safeNext || (data.user.role === 'admin' ? 'admin.html' : 'prediction.html'); }
  catch(error){ status.textContent = error.message; }
});
if (registerForm) registerForm.addEventListener('submit', async (event) => {
  event.preventDefault(); const status = document.getElementById('register-status');
  const body = Object.fromEntries(new FormData(registerForm));
  try { const response = await fetch('/api/auth/register', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const data = await response.json(); if(!response.ok) throw new Error(data.detail || 'Registration failed'); status.textContent = `Account created for ${data.username}. You can now sign in.`; registerForm.reset(); }
  catch(error){ status.textContent = error.message; }
});
