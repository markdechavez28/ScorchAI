const MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"];
const WEATHER_FIELDS = ["MinTemp","MaxTemp","Rainfall","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","WindSpeed9am","WindSpeed3pm","Sunshine","Cloud9am","Cloud3pm"];

let LOCATIONS = [];

const TAB_META = {
  chat: "Chat Agent",
  predict: "Predict Output",
  farm: "Farm Sizing",
  bestmonth: "Best Month",
  cloud: "Cloud Sensitivity",
  map: "City Explorer",
};

// ---------- tabs ----------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    document.getElementById("page-title").textContent = TAB_META[btn.dataset.tab];
    document.getElementById("crumb-current").textContent = TAB_META[btn.dataset.tab];
    if (btn.dataset.tab === "map") setTimeout(() => map && map.invalidateSize(), 50);
  });
});

// ---------- shared form helpers ----------
function populateMonthSelects() {
  const now = new Date().getMonth() + 1;
  document.querySelectorAll(".month-select").forEach(sel => {
    sel.innerHTML = MONTH_NAMES.map((name, i) =>
      `<option value="${i + 1}" ${i + 1 === now ? "selected" : ""}>${name}</option>`).join("");
  });
}

function populateLocationSelects() {
  document.querySelectorAll(".location-select").forEach(sel => {
    sel.innerHTML = LOCATIONS.map(loc => `<option value="${loc.code}">${loc.display_name}</option>`).join("");
  });
}

function buildWeatherFields() {
  document.querySelectorAll(".weather-fields").forEach(container => {
    container.innerHTML = WEATHER_FIELDS.map(f =>
      `<label>${f} <input type="number" step="any" name="weather_${f}" placeholder="auto"></label>`).join("");
  });
}

function collectWeather(form) {
  const weather = {};
  WEATHER_FIELDS.forEach(f => {
    const el = form.querySelector(`[name="weather_${f}"]`);
    if (el && el.value !== "") weather[f] = parseFloat(el.value);
  });
  return Object.keys(weather).length ? weather : null;
}

async function postJSON(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return res.json();
}
async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

// ---------- result card rendering ----------
function renderPredictResult(card, r) {
  if (r.error) { card.innerHTML = `<p class="error">${r.error}</p>`; return; }
  const assumed = r.assumed_climatological_inputs.length
    ? `<p class="disclosure">Assumed from ${r.location}'s historical average for ${MONTH_NAMES[r.month - 1]}: ${r.assumed_climatological_inputs.join(", ")}.</p>`
    : "";
  card.innerHTML = `
    <p class="stat-label">Expected output &mdash; ${r.location}, ${MONTH_NAMES[r.month - 1]}</p>
    <span class="big-number">${r.predicted_kWh_per_kWp_per_day.toFixed(2)}<span class="unit">kWh/kWp/day</span></span>
    <div class="badge-row">
      <span class="badge">${r.model_variant} model</span>
      <span class="badge teal">R&sup2; = ${r.model_holdout_r2}</span>
    </div>
    <div class="stat-row">
      <div class="stat"><strong>${r.confidence_range_kWh_per_kWp_per_day[0]}</strong><span>low (1&sigma;)</span></div>
      <div class="stat"><strong>${r.confidence_range_kWh_per_kWp_per_day[1]}</strong><span>high (1&sigma;)</span></div>
    </div>
    ${assumed}`;
}

function renderFarmResult(card, r) {
  if (r.error) { card.innerHTML = `<p class="error">${r.error}</p>`; return; }
  const assumed = r.assumed_climatological_inputs.length
    ? `<p class="disclosure">Assumed from ${r.location}'s historical average for ${MONTH_NAMES[r.month - 1]}: ${r.assumed_climatological_inputs.join(", ")}.</p>`
    : "";
  card.innerHTML = `
    <p class="stat-label">Estimated output &mdash; ${r.location}, ${MONTH_NAMES[r.month - 1]}</p>
    <span class="big-number">${r.estimated_output_kWh.toLocaleString()}<span class="unit">kWh</span></span>
    <div class="badge-row">
      <span class="badge">${r.installed_capacity_kW.toLocaleString()} kW</span>
      <span class="badge">${r.days} day(s)</span>
      <span class="badge teal">${r.model_variant} model, R&sup2; = ${r.model_holdout_r2}</span>
    </div>
    <div class="stat-row">
      <div class="stat"><strong>${r.estimated_output_range_kWh[0].toLocaleString()}</strong><span>low (1&sigma;)</span></div>
      <div class="stat"><strong>${r.estimated_output_range_kWh[1].toLocaleString()}</strong><span>high (1&sigma;)</span></div>
    </div>
    ${assumed}`;
}

// ---------- chat ----------
const chatLog = document.getElementById("chat-log");
const chatEmptyState = document.getElementById("chat-empty-state");
let activeConversationId = null;
let CONVERSATIONS = [];

function showChatLog() {
  chatEmptyState.style.display = "none";
  chatLog.style.display = "block";
}
function showChatEmptyState() {
  chatLog.innerHTML = "";
  chatLog.style.display = "none";
  chatEmptyState.style.display = "block";
}

function renderMarkdown(text) {
  // Agent answers (and replayed history) are markdown; render it, but sanitize the
  // resulting HTML in case the model ever echoes back something injection-like.
  if (typeof marked === "undefined") return text;
  const html = marked.parse(text, { breaks: true });
  return typeof DOMPurify !== "undefined" ? DOMPurify.sanitize(html) : html;
}

function addChatMsg(who, text) {
  const div = document.createElement("div");
  div.className = "chat-msg " + (who === "You" ? "user" : "agent");
  div.innerHTML = `<div class="bubble">${renderMarkdown(text)}</div>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}
async function askAgent(question) {
  showChatLog();
  addChatMsg("You", question);
  const r = await postJSON("/api/chat", { question, conversation_id: activeConversationId });
  addChatMsg("Agent", r.error || r.answer);
  if (r.conversation_id && r.conversation_id !== activeConversationId) {
    activeConversationId = r.conversation_id;
    await loadConversations();
  }
}
document.getElementById("chat-form").addEventListener("submit", e => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  if (!input.value.trim()) return;
  askAgent(input.value.trim());
  input.value = "";
});
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => askAgent(btn.textContent));
});

// ---------- conversation history (sidebar, like Claude desktop) ----------
function renderConversationList() {
  const list = document.getElementById("conversation-list");
  if (!CONVERSATIONS.length) {
    list.innerHTML = `<li class="empty-hint">No past conversations yet.</li>`;
    return;
  }
  list.innerHTML = CONVERSATIONS.map(c => `
    <li data-id="${c.id}" class="${c.id === activeConversationId ? "active" : ""}">
      <span class="conv-title">${c.title}</span>
      <button type="button" class="conv-delete" data-id="${c.id}" title="Delete">&times;</button>
    </li>`).join("");
  list.querySelectorAll("li[data-id]").forEach(li => {
    li.addEventListener("click", () => selectConversation(parseInt(li.dataset.id, 10)));
  });
  list.querySelectorAll(".conv-delete").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      deleteConversation(parseInt(btn.dataset.id, 10));
    });
  });
}

async function loadConversations() {
  CONVERSATIONS = await getJSON("/api/conversations");
  renderConversationList();
}

async function selectConversation(id) {
  const messages = await getJSON(`/api/conversations/${id}/messages`);
  activeConversationId = id;
  showChatLog();
  chatLog.innerHTML = "";
  messages.forEach(m => addChatMsg(m.role === "user" ? "You" : "Agent", m.content));
  renderConversationList();
}

function startNewChat() {
  activeConversationId = null;
  showChatEmptyState();
  renderConversationList();
}
document.getElementById("new-chat-btn").addEventListener("click", startNewChat);

async function deleteConversation(id) {
  if (!confirm("Delete this conversation?")) return;
  await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  if (id === activeConversationId) startNewChat();
  await loadConversations();
}

// ---------- predict ----------
document.getElementById("predict-form").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  const body = {
    location: form.location.value,
    month: parseInt(form.month.value, 10),
    weather: collectWeather(form),
  };
  const r = await postJSON("/api/predict", body);
  renderPredictResult(document.getElementById("predict-result"), r);
});

// ---------- farm ----------
const farmForm = document.getElementById("farm-form");
farmForm.querySelectorAll('input[name="sizing_mode"]').forEach(radio => {
  radio.addEventListener("change", () => {
    const byCapacity = farmForm.sizing_mode.value === "capacity";
    document.getElementById("capacity-label").style.display = byCapacity ? "" : "none";
    document.getElementById("area-label").style.display = byCapacity ? "none" : "";
  });
});
farmForm.addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  const byCapacity = form.sizing_mode.value === "capacity";
  const body = {
    location: form.location.value,
    month: parseInt(form.month.value, 10),
    capacity_kw: byCapacity ? parseFloat(form.capacity_kw.value) : null,
    area_m2: byCapacity ? null : parseFloat(form.area_m2.value),
    weather: collectWeather(form),
    days: parseInt(form.days.value, 10) || 1,
  };
  const r = await postJSON("/api/farm", body);
  renderFarmResult(document.getElementById("farm-result"), r);
});

// ---------- best month chart ----------
let bestMonthChart = null;
document.getElementById("bestmonth-form").addEventListener("submit", async e => {
  e.preventDefault();
  const location = e.target.location.value;
  const r = await getJSON(`/api/best_month?location=${encodeURIComponent(location)}`);
  if (r.error) { alert(r.error); return; }
  const sorted = [...r.ranking].sort((a, b) => a.month - b.month);
  const ctx = document.getElementById("bestmonth-chart");
  if (bestMonthChart) bestMonthChart.destroy();
  bestMonthChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(row => MONTH_NAMES[row.month - 1]),
      datasets: [{ label: `${r.location}: PVOUT baseline (kWh/kWp/day)`, data: sorted.map(row => row.PVOUT_avg_daily), backgroundColor: "#f6c945", borderRadius: 6 }],
    },
    options: { scales: { y: { beginAtZero: true, grid: { color: "#e7e8ee" } }, x: { grid: { display: false } } }, plugins: { legend: { labels: { color: "#15161a" } } } },
  });
});

// ---------- cloud sensitivity chart ----------
let cloudChart = null;
document.getElementById("cloud-form").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  const location = form.location.value;
  const month = form.month.value;
  const r = await getJSON(`/api/cloud_sensitivity?location=${encodeURIComponent(location)}&month=${month}`);
  if (r.error) { alert(r.error); return; }
  const oktas = Object.keys(r.output_by_cloud_oktas);
  const ctx = document.getElementById("cloud-chart");
  if (cloudChart) cloudChart.destroy();
  cloudChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: oktas.map(o => `${o} oktas`),
      datasets: [
        { label: "Output (kWh/kWp/day)", data: oktas.map(o => r.output_by_cloud_oktas[o]), backgroundColor: "#f6c945", borderRadius: 6, yAxisID: "y" },
        { label: "% reduction vs clear", data: oktas.map(o => r.pct_reduction_vs_clear[o]), type: "line", borderColor: "#3f9c84", backgroundColor: "#3f9c84", tension: 0.3, yAxisID: "y1" },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "#15161a" } } },
      scales: {
        y: { beginAtZero: true, position: "left", title: { display: true, text: "kWh/kWp/day" }, grid: { color: "#e7e8ee" } },
        y1: { beginAtZero: true, position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "% reduction" } },
        x: { grid: { display: false } },
      },
    },
  });
});

// ---------- map ----------
let map = null;
function initMap() {
  map = L.map("map").setView([-25.5, 134], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
  const values = LOCATIONS.map(l => l.pvout_annual_avg_daily);
  const min = Math.min(...values), max = Math.max(...values);
  LOCATIONS.forEach(loc => {
    const t = (loc.pvout_annual_avg_daily - min) / (max - min || 1);
    const radius = 5 + t * 10;
    // low output -> teal, high output -> amber
    const color = t < 0.5
      ? `rgb(${Math.round(63 + t * 2 * 192)}, ${Math.round(156 + t * 2 * 45)}, ${Math.round(132 - t * 2 * 61)})`
      : `rgb(${246}, ${Math.round(201 - (t - 0.5) * 2 * 60)}, ${Math.round(69 - (t - 0.5) * 2 * 69)})`;
    L.circleMarker([loc.lat, loc.lon], { radius, color, fillColor: color, fillOpacity: 0.8 })
      .addTo(map)
      .bindPopup(`<strong>${loc.display_name}</strong><br>Annual PVOUT: ${loc.pvout_annual_avg_daily} kWh/kWp/day`);
  });
}

// ---------- auth ----------
let authMode = "login"; // "login" | "signup"

function showAuthScreen() {
  document.getElementById("auth-screen").style.display = "flex";
  document.getElementById("shell").style.display = "none";
}

function setAuthMode(mode) {
  authMode = mode;
  document.getElementById("mode-login").classList.toggle("active", mode === "login");
  document.getElementById("mode-signup").classList.toggle("active", mode === "signup");
  document.getElementById("auth-submit").textContent = mode === "login" ? "Log in" : "Create account";
  document.getElementById("auth-error").style.display = "none";
}
document.getElementById("mode-login").addEventListener("click", () => setAuthMode("login"));
document.getElementById("mode-signup").addEventListener("click", () => setAuthMode("signup"));

document.getElementById("auth-form").addEventListener("submit", async e => {
  e.preventDefault();
  const username = document.getElementById("auth-username").value.trim();
  const password = document.getElementById("auth-password").value;
  const url = authMode === "login" ? "/api/login" : "/api/signup";
  const r = await postJSON(url, { username, password });
  const errEl = document.getElementById("auth-error");
  if (r.error) {
    errEl.textContent = r.error;
    errEl.style.display = "block";
    return;
  }
  errEl.style.display = "none";
  await initApp(r.username);
});

document.getElementById("logout-btn").addEventListener("click", async () => {
  await postJSON("/api/logout", {});
  LOCATIONS = [];
  CONVERSATIONS = [];
  activeConversationId = null;
  chatLog.innerHTML = "";
  document.getElementById("conversation-list").innerHTML = "";
  if (bestMonthChart) { bestMonthChart.destroy(); bestMonthChart = null; }
  if (cloudChart) { cloudChart.destroy(); cloudChart = null; }
  document.getElementById("auth-form").reset();
  showAuthScreen();
});

async function initApp(username) {
  document.getElementById("auth-screen").style.display = "none";
  document.getElementById("shell").style.display = "flex";
  document.getElementById("user-name").textContent = username;
  populateMonthSelects();
  buildWeatherFields();
  LOCATIONS = await getJSON("/api/locations");
  populateLocationSelects();
  initMap();
  await loadConversations();
  if (CONVERSATIONS.length) {
    await selectConversation(CONVERSATIONS[0].id);
  } else {
    startNewChat();
  }
}

// ---------- init ----------
(async function checkAuth() {
  const me = await getJSON("/api/me");
  if (me && me.username) {
    await initApp(me.username);
  } else {
    showAuthScreen();
  }
})();
