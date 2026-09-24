"use strict";

// This script supplies browser-only behavior to source-derived GreggSpeak markup.
// All page data is defined in mock-data.js; no backend endpoint is contacted.
const STORAGE_KEY = "greggspeak-source-ui-demo-v1";
const app = document.getElementById("app");
const toast = document.getElementById("demoToast");
let toastTimer;
let processingId = null;
let pendingConfirmation = null;

function freshState() { return { edits: {}, fields: {}, processed: {}, archived: [] }; }
function loadState() {
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        if (saved && typeof saved === "object") return {
            edits: saved.edits && typeof saved.edits === "object" ? saved.edits : {},
            fields: saved.fields && typeof saved.fields === "object" ? saved.fields : {},
            processed: saved.processed && typeof saved.processed === "object" ? saved.processed : {},
            archived: Array.isArray(saved.archived) ? saved.archived : []
        };
    } catch (_) { /* In-memory state remains usable when local storage is blocked. */ }
    return freshState();
}
let state = loadState();
function saveState() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
    catch (_) { /* This does not affect the current tab's interactions. */ }
}
function h(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);
}
function findBatch(id) { return DEMO_BATCHES.find((batch) => batch.id === id); }
function archived(id) { return state.archived.includes(id); }
function processed(batch) { return batch.processed || state.processed[batch.id] === true; }
function average(batch) { return (batch.pages.reduce((total, page) => total + page.confidence, 0) / batch.pages.length).toFixed(2); }
function textFor(batch, page) {
    const edited = state.edits[`${batch.id}:${page.number}`];
    return typeof edited === "string" ? edited : page.text;
}
function combined(batch) { return processed(batch) ? batch.pages.map((page) => textFor(batch, page)).join("\n\n") : ""; }
function fieldsFor(batch) {
    return {
        header_line_1: batch.court, header_line_2: "Transcript of Stenographic Notes", header_line_3: "", header_line_4: "",
        email: "", mobile: "", complainant: "", criminal_case_no: batch.caseNo, case_for: "", accused: "",
        proceeding: "Hearing", hearing_date: batch.hearingDate || "", hearing_time: "", hearing_period: "morning",
        place: batch.place || "", judge_name: "", public_prosecutor: "", defense_counsel: "",
        court_interpreter: "", court_stenographer: "", certifier_name: "", order_text: "",
        ...(state.fields[batch.id] || {})
    };
}
function notify(message) {
    toast.textContent = message;
    toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add("hidden"), 3200);
}
function route() {
    const parts = location.hash.slice(1).split("/").filter(Boolean);
    if (parts[0] === "records" || parts[0] === "trash") return { mode: parts[0] };
    if (parts[0] === "batch" && findBatch(parts[1])) return { mode: "batch", id: parts[1], page: Number(parts[3]) || 1 };
    return { mode: "home" };
}
function closeMenus() {
    document.getElementById("primaryNav").classList.remove("is-open");
    document.getElementById("mobileMenuButton").classList.remove("is-open");
    document.getElementById("mobileMenuButton").setAttribute("aria-expanded", "false");
    document.getElementById("profileMenuList").classList.add("hidden");
    document.getElementById("profileMenuButton").setAttribute("aria-expanded", "false");
}
function navState(mode) {
    document.querySelectorAll("[data-nav]").forEach((link) => {
        const active = link.dataset.nav === (mode === "batch" ? "records" : mode);
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    });
    closeMenus();
}
function hero(eyebrow, title, description, label, status, detail) {
    return `<section class="hero"><div class="hero-copy"><p class="eyebrow">${eyebrow}</p><h1>${h(title)}</h1><p class="subtle">${description}</p></div><div class="hero-card"><span>${label}</span><strong>${h(status)}</strong><p>${h(detail)}</p></div></section>`;
}
function meta(items) {
    return `<div class="meta-grid">${items.map(([label, value]) => `<div><span>${h(label)}</span><strong>${h(value)}</strong></div>`).join("")}</div>`;
}

function renderHome() {
    const batches = DEMO_BATCHES.filter((batch) => !archived(batch.id));
    const latest = batches[0];
    app.innerHTML = hero("Stenographer Workspace", "Latest Transcript Overview", "Review the most recent LCD transcript, reopen the current workspace quickly, and monitor the transcript archive at a glance.", "Workspace Status", "Ready for Review", `${batches.length} active batch${batches.length === 1 ? "" : "es"} available`) +
        `<section class="card access-card"><div class="section-head"><div><p class="eyebrow">Network Access</p><h2>Open GreggSpeak From Another Device</h2></div></div><div class="access-grid"><div class="access-main"><label for="primaryAccessUrl">Primary URL</label><div class="copy-row"><input id="primaryAccessUrl" class="search-input" type="text" value="Local device connection unavailable" readonly><button class="btn primary small demo-muted-action" type="button" disabled>Copy Link</button></div></div><div class="qr-card"><div class="demo-qr-placeholder" aria-label="Device QR access unavailable">GS</div><span>Local device access unavailable.</span></div></div></section>` +
        (latest ? `<section id="latest-batch" class="card latest-card"><div class="section-head"><div><p class="eyebrow">Latest Batch</p><h2>${h(latest.title)}</h2></div><div class="action-row"><a class="btn primary" href="#batch/${latest.id}/page/1">Open Workspace</a></div></div>${meta([["Date", latest.date], ["Pages", latest.pages.length], ["Confidence", processed(latest) ? average(latest) : "—"], ["Batch ID", latest.id]])}</section>` : `<section class="card"><p class="empty-state">No transcript batches are available yet.</p></section>`) +
        `<section class="card dashboard-summary-card"><div class="section-head"><div><p class="eyebrow">Quick Summary</p><h2>Archive Snapshot</h2></div></div>${meta([["Active Records", batches.length], ["Deleted Records", state.archived.length], ["Latest Batch ID", latest?.id || "Unavailable"], ["Latest Title", latest?.title || "No saved transcript"]])}<div class="action-row"><a class="btn primary" href="#records">Open Records</a><a class="btn secondary" href="#trash">Open Trash Bin</a></div></section>`;
}

function recordRow(batch, isTrash) {
    return `<tr data-search="${h(`${batch.id} ${batch.title}`.toLowerCase())}"><td><span class="mono-text">${h(batch.id)}</span></td><td>${h(batch.title)}</td><td>${h(batch.date)}</td><td>${batch.pages.length}</td>${isTrash ? "" : `<td>${processed(batch) ? average(batch) : "—"}</td>`}<td><span class="status-tag ${isTrash ? "deleted" : processed(batch) ? "processed" : "pending"}">${isTrash ? "deleted" : processed(batch) ? "processed" : "pending"}</span></td><td><div class="action-row">${isTrash ? `<button class="btn secondary small" type="button" data-action="restore" data-batch="${batch.id}">Restore</button><button class="btn danger small" type="button" disabled title="Permanent deletion is unavailable">Delete</button>` : `<a class="btn primary small" href="#batch/${batch.id}/page/1">Open</a><button class="btn danger small" type="button" data-action="archive" data-batch="${batch.id}">Delete</button>`}</div></td></tr>`;
}
function renderRecords(isTrash) {
    const batches = DEMO_BATCHES.filter((batch) => archived(batch.id) === isTrash);
    app.innerHTML = hero(isTrash ? "Archive Recovery" : "Transcript Archive", isTrash ? "Deleted Transcript Records" : "Saved Transcript Records", isTrash ? "Review transcript batches that were removed from the active archive and restore them when needed." : "Browse all active transcript batches, search by title or batch ID, and reopen any saved workspace.", isTrash ? "Trash Bin Status" : "Archive Status", `${batches.length} ${isTrash ? "Deleted" : "Active"} Record${batches.length === 1 ? "" : "s"}`, isTrash ? "Deleted transcript batches remain recoverable." : "Use search to locate saved transcript batches faster.") +
        `<section id="${isTrash ? "trash-bin" : "saved-records"}" class="card"><div class="section-head ${isTrash ? "" : "section-head-stack"}"><div><p class="eyebrow">${isTrash ? "Trash Bin" : "Saved Records"}</p><h2>${isTrash ? "Deleted Transcripts" : "All Transcript Batches"}</h2></div>${isTrash ? `<span class="toolbar-count">${batches.length} deleted</span>` : `<div class="toolbar"><input id="batchSearch" class="search-input" type="search" placeholder="Search by title or batch ID" aria-label="Search batches"></div>`}</div><div class="table-wrap"><table><thead><tr><th>Batch ID</th><th>Title</th><th>Date</th><th>Pages</th>${isTrash ? "" : "<th>Confidence</th>"}<th>Status</th><th>Actions</th></tr></thead><tbody id="batchTableBody">${batches.map((batch) => recordRow(batch, isTrash)).join("")}</tbody></table></div><p id="searchEmptyState" class="empty-state ${batches.length ? "hidden" : ""}">${isTrash ? "Trash bin is empty." : "No active transcripts have been saved yet."}</p></section>`;
}

function field(name, label, value, type = "text") {
    return `<label>${label}<input name="${name}" type="${type}" maxlength="100" value="${h(value)}"></label>`;
}
function tsnForm(batch, fields) {
    const period = ["morning", "afternoon", "evening"].map((name) => `<option value="${name}" ${fields.hearing_period === name ? "selected" : ""}>${name[0].toUpperCase() + name.slice(1)}</option>`).join("");
    return `<form id="tsnExportForm" class="tsn-export-form" data-batch="${batch.id}"><div class="section-head compact-head"><div><p class="eyebrow">TSN Export Fields</p><h2>Front And Certification Pages</h2></div><span class="toolbar-count">Manual fields for export</span></div><div class="tsn-scroll-panel"><div class="tsn-form-section"><h3>Header / Contact</h3><div class="tsn-field-grid">${field("header_line_1", "Header Line 1", fields.header_line_1)}${field("header_line_2", "Header Line 2", fields.header_line_2)}${field("header_line_3", "Header Line 3", fields.header_line_3)}${field("header_line_4", "Header Line 4", fields.header_line_4)}${field("email", "Email Add", fields.email)}${field("mobile", "Mobile Number", fields.mobile)}</div></div><div class="tsn-form-section"><h3>Case Details</h3><div class="tsn-field-grid">${field("complainant", "Complainant / Plaintiff", fields.complainant)}${field("criminal_case_no", "Criminal Case No.", fields.criminal_case_no)}${field("case_for", "For", fields.case_for)}${field("accused", "Accused", fields.accused)}${field("proceeding", "Proceeding", fields.proceeding)}${field("hearing_date", "Hearing Date", fields.hearing_date, "date")}${field("hearing_time", "Hearing Time", fields.hearing_time, "time")}<label>Time Period<select name="hearing_period">${period}</select></label>${field("place", "Place", fields.place)}${field("judge_name", "Presiding Judge", fields.judge_name)}</div></div><div class="tsn-form-section"><h3>Appearances / Present</h3><div class="tsn-field-grid">${field("public_prosecutor", "Public Prosecutor", fields.public_prosecutor)}${field("defense_counsel", "Counsel for the Accused", fields.defense_counsel)}${field("court_interpreter", "Court Interpreter", fields.court_interpreter)}${field("court_stenographer", "Court Stenographer", fields.court_stenographer)}${field("certifier_name", "Certifier Name", fields.certifier_name)}</div><label class="tsn-wide-field">Final Court Order / Notes<textarea name="order_text" maxlength="3000" placeholder="Optional text after COURT: Order.">${h(fields.order_text)}</textarea></label></div></div><div class="editor-actions"><button class="btn secondary" type="submit">Save TSN Details</button><button class="btn primary" type="button" disabled title="PDF export requires the real application">Download TSN PDF</button><button class="btn secondary" type="button" disabled title="Word export requires the real application">Download Editable Word</button></div></form>`;
}
function pageCard(batch, page, selected) {
    const ready = processed(batch);
    const edited = typeof state.edits[`${batch.id}:${page.number}`] === "string";
    return `<article id="page-${page.number}" class="page-view user-edit-card ${edited ? "page-editor-edited" : ""}" ${selected ? 'data-selected="true"' : ""}><div class="page-meta"><strong>Page ${page.number}</strong><span>Confidence ${ready ? page.confidence.toFixed(2) : "—"}</span></div><div class="review-metrics"><span>Rows ${ready ? page.rows : 0}</span><span>Words ${ready ? page.words : 0}</span><span>Review Words ${ready ? page.reviewWords : 0}</span></div>${edited ? `<span class="edited-badge">Edited</span>` : ""}${ready && page.reviewWords ? `<div class="review-alert"><strong>Review Suggested</strong><span>Some words may need manual review.</span></div>` : ""}<div class="segmentation-review-grid"><figure class="segmentation-review-panel"><figcaption>Scanned Page</figcaption><a href="${h(page.image || "./assets/synthetic-page.svg")}" target="_blank" rel="noopener"><img src="${h(page.image || "./assets/synthetic-page.svg")}" alt="${page.image ? "Scanned shorthand page" : "Illustrative shorthand page"}"></a></figure><figure class="segmentation-review-panel"><figcaption>Segmentation Review</figcaption><a href="${h(page.segmentationImage || "./assets/synthetic-segmentation.svg")}" target="_blank" rel="noopener"><img src="${h(page.segmentationImage || "./assets/synthetic-segmentation.svg")}" alt="${page.segmentationImage ? "Saved segmentation review" : "Illustrative segmentation review"}"></a></figure></div>${ready ? `<form class="demo-page-form" data-batch="${batch.id}" data-page="${page.number}"><label for="user_edited_text_${batch.id}_${page.number}">Transcript Text</label><textarea id="user_edited_text_${batch.id}_${page.number}" name="edited_text" maxlength="20000">${h(textFor(batch, page))}</textarea><div class="editor-actions"><button class="btn primary" type="submit">Save Changes</button></div></form>` : `<div class="review-alert"><strong>Recognition Pending</strong><span>Load the available page result to continue reviewing.</span></div><button class="btn primary" type="button" data-action="simulate" data-batch="${batch.id}" ${processingId === batch.id ? "disabled" : ""}>${processingId === batch.id ? "Loading result…" : "Load Example Result"}</button>`}<div class="editor-actions"><button class="btn danger" type="button" disabled title="Page deletion is unavailable">Delete Page</button></div></article>`;
}
function renderBatch(id, selectedPage) {
    const batch = findBatch(id);
    if (archived(id)) { location.hash = "#trash"; return; }
    const ready = processed(batch);
    const transcript = combined(batch);
    const fields = fieldsFor(batch);
    app.innerHTML = hero("Transcript Workspace", batch.title, "Review, edit, and export this transcript batch from one stenographer-friendly workspace.", "Batch Status", ready ? "processed" : "pending", `${batch.pages.length} page(s)`) +
        `<section class="detail-shell user-detail-shell"><article class="card transcript-reader-card"><div class="section-head"><div><p class="eyebrow">Transcript</p><h2>Combined Output</h2></div><div class="action-row"><button id="ttsToggleButton" class="btn secondary small" type="button" data-action="speak" ${ready ? "" : "disabled"}>Play Audio</button><button class="btn secondary small" type="submit" form="tsnExportForm">Save TSN Details</button><button class="btn primary small" type="button" disabled title="PDF export requires the real application">Download PDF</button><button class="btn secondary small" type="button" disabled title="Word export requires the real application">Download Word</button><button class="btn secondary small" type="button" disabled title="Hardware scanning is not connected">Add Pages</button><a class="btn workspace-back-btn small" href="#home">Back</a></div></div>${meta([["Batch ID", batch.id], ["Date", batch.date], ["Pages", batch.pages.length], ["Confidence", ready ? average(batch) : "—"]])}${tsnForm(batch, fields)}<div class="section-head transcript-output-head"><div><p class="eyebrow">Recognized Transcript</p><h2>TSN Body</h2></div></div><div class="transcript-panel transcript-reader"><pre>${ready ? h(transcript) : "Transcript not generated yet."}</pre></div></article><aside class="card"><div class="section-head"><h2>Editable Pages</h2><div class="action-row"><span class="toolbar-count">${batch.pages.length} page(s)</span><button class="btn danger small" type="button" data-action="archive" data-batch="${batch.id}">Delete Batch</button></div></div><nav class="demo-page-nav" aria-label="Choose page">${batch.pages.map((page) => `<a class="${page.number === selectedPage ? "active" : ""}" href="#batch/${batch.id}/page/${page.number}" ${page.number === selectedPage ? 'aria-current="page"' : ""}>Page ${page.number}</a>`).join("")}</nav><div class="page-view-list">${batch.pages.map((page) => pageCard(batch, page, page.number === selectedPage)).join("")}</div></aside></section>`;
    const selected = document.getElementById(`page-${selectedPage}`);
    if (selected) selected.scrollIntoView({ block: "nearest" });
}
function render() {
    const current = route();
    navState(current.mode);
    if (current.mode === "home") renderHome();
    else if (current.mode === "records") renderRecords(false);
    else if (current.mode === "trash") renderRecords(true);
    else renderBatch(current.id, current.page);
}

function openConfirm(title, message, confirmLabel, action) {
    pendingConfirmation = action;
    document.getElementById("confirmTitle").textContent = title;
    document.getElementById("confirmMessage").textContent = message;
    document.getElementById("confirmSubmitButton").textContent = confirmLabel;
    document.getElementById("confirmModal").classList.remove("hidden");
    document.getElementById("confirmModal").setAttribute("aria-hidden", "false");
}
function closeConfirm() {
    pendingConfirmation = null;
    document.getElementById("confirmModal").classList.add("hidden");
    document.getElementById("confirmModal").setAttribute("aria-hidden", "true");
}

document.addEventListener("submit", (event) => {
    const form = event.target;
    if (form.classList.contains("demo-page-form")) {
        event.preventDefault();
        state.edits[`${form.dataset.batch}:${form.dataset.page}`] = form.elements.edited_text.value.slice(0, 20000);
        saveState(); render(); notify("Page changes saved in this browser. Combined text updated.");
    } else if (form.id === "tsnExportForm") {
        event.preventDefault();
        const data = Object.fromEntries(new FormData(form).entries());
        state.fields[form.dataset.batch] = Object.fromEntries(Object.entries(data).map(([key, value]) => [key, String(value).slice(0, key === "order_text" ? 3000 : 100)]));
        saveState(); render(); notify("TSN details saved.");
    }
});
document.addEventListener("input", (event) => {
    if (event.target.id !== "batchSearch") return;
    const query = event.target.value.trim().toLowerCase();
    let visible = 0;
    document.querySelectorAll("#batchTableBody tr").forEach((row) => {
        row.style.display = row.dataset.search.includes(query) ? "" : "none";
        if (row.style.display !== "none") visible += 1;
    });
    const empty = document.getElementById("searchEmptyState");
    empty.classList.toggle("hidden", visible !== 0);
    empty.textContent = query ? "No batches match your search." : "No active transcripts have been saved yet.";
});
document.addEventListener("click", (event) => {
    if (event.target.closest("#mobileMenuButton")) {
        const nav = document.getElementById("primaryNav");
        const button = document.getElementById("mobileMenuButton");
        const open = nav.classList.toggle("is-open");
        button.classList.toggle("is-open", open);
        button.setAttribute("aria-expanded", String(open));
        return;
    }
    if (event.target.closest("#profileMenuButton")) {
        const menu = document.getElementById("profileMenuList");
        const open = menu.classList.toggle("hidden") === false;
        document.getElementById("profileMenuButton").setAttribute("aria-expanded", String(open));
        return;
    }
    if (event.target.closest("#resetDemoButton")) {
        closeMenus();
        openConfirm("Reset Data?", "Restore the original records and clear saved edits?", "Reset Data", () => {
            state = freshState(); saveState(); location.hash = "#home"; render(); notify("Data reset.");
        });
        return;
    }
    if (event.target.closest("#confirmSubmitButton")) {
        const action = pendingConfirmation; closeConfirm(); if (action) action(); return;
    }
    if (event.target.closest("#confirmCancelButton") || event.target.closest("[data-confirm-close]")) { closeConfirm(); return; }
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    const id = button.dataset.batch;
    if (action === "simulate" && findBatch(id) && !processingId) {
        processingId = id; render();
        setTimeout(() => { state.processed[id] = true; processingId = null; saveState(); render(); notify("Page result loaded."); }, 850);
    } else if (action === "archive" && findBatch(id)) {
        openConfirm("Delete Transcript?", "Move this transcript batch to the trash bin?", "Delete", () => {
            if (!archived(id)) state.archived.push(id);
            saveState(); location.hash = "#records"; render(); notify(`${id} moved to Trash.`);
        });
    } else if (action === "restore" && findBatch(id)) {
        openConfirm("Restore Transcript?", "Restore this transcript batch from Trash?", "Restore", () => {
            state.archived = state.archived.filter((value) => value !== id);
            saveState(); render(); notify(`${id} restored to Records.`);
        });
    } else if (action === "speak") {
        if (!("speechSynthesis" in window)) { notify("Browser audio is unavailable."); return; }
        if (speechSynthesis.speaking) { speechSynthesis.cancel(); notify("Audio stopped."); return; }
        const batch = findBatch(route().id);
        if (batch && processed(batch)) { speechSynthesis.speak(new SpeechSynthesisUtterance(combined(batch))); notify("Playing transcript audio."); }
    }
});
document.addEventListener("keydown", (event) => { if (event.key === "Escape") { closeConfirm(); closeMenus(); } });
window.addEventListener("hashchange", () => { if ("speechSynthesis" in window) speechSynthesis.cancel(); render(); window.scrollTo(0, 0); });
render();
