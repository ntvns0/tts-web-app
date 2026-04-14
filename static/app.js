const form = document.getElementById("tts-form");
const fileInput = document.getElementById("file-input");
const titleInput = document.getElementById("title-input");
const textInput = document.getElementById("text-input");
const clearButton = document.getElementById("clear-button");
const dropzone = document.getElementById("dropzone");
const selectedFile = document.getElementById("selected-file");
const voiceSelect = document.getElementById("voice-select");
const voiceDescription = document.getElementById("voice-description");
const jobList = document.getElementById("job-list");
const jobCount = document.getElementById("job-count");
const librarySearchInput = document.getElementById("library-search");
const libraryFilterSelect = document.getElementById("library-filter");
const librarySortSelect = document.getElementById("library-sort");
const activeJobList = document.getElementById("active-job-list");
const queuePanelTitle = document.getElementById("queue-panel-title");
const queuePanelCount = document.getElementById("queue-panel-count");
const systemPanelTitle = document.getElementById("system-panel-title");
const systemPanelUpdated = document.getElementById("system-panel-updated");
const cpuBusyPill = document.getElementById("cpu-busy-pill");
const cpuSummary = document.getElementById("cpu-summary");
const cpuMeterFill = document.getElementById("cpu-meter-fill");
const cpuBreakdown = document.getElementById("cpu-breakdown");
const gpuBusyPill = document.getElementById("gpu-busy-pill");
const gpuSummary = document.getElementById("gpu-summary");
const gpuMeterFill = document.getElementById("gpu-meter-fill");
const gpuBreakdown = document.getElementById("gpu-breakdown");

const audioElement = document.getElementById("audio-element");
const playButton = document.getElementById("play-button");
const pauseButton = document.getElementById("pause-button");
const backButton = document.getElementById("back-button");
const forwardButton = document.getElementById("forward-button");
const timeline = document.getElementById("timeline");
const currentTimeEl = document.getElementById("current-time");
const durationTimeEl = document.getElementById("duration-time");
const trackTitle = document.getElementById("track-title");
const trackVoice = document.getElementById("track-voice");
const playerSubtitle = document.getElementById("player-subtitle");
const downloadLink = document.getElementById("download-link");
const transcriptViewer = document.getElementById("transcript-viewer");
const sectionNav = document.getElementById("section-nav");
const sectionCount = document.getElementById("section-count");
const readerMode = document.getElementById("reader-mode");
const syncOffsetInput = document.getElementById("sync-offset");
const syncDriftInput = document.getElementById("sync-drift");
const syncOffsetValue = document.getElementById("sync-offset-value");
const syncDriftValue = document.getElementById("sync-drift-value");
const syncEarlierButton = document.getElementById("sync-earlier");
const syncLaterButton = document.getElementById("sync-later");
const driftLessButton = document.getElementById("drift-less");
const driftMoreButton = document.getElementById("drift-more");
const syncResetButton = document.getElementById("sync-reset");
const followPlaybackInput = document.getElementById("follow-playback");
const calibrateStartButton = document.getElementById("calibrate-start");
const calibrateEndButton = document.getElementById("calibrate-end");
const clearCalibrationButton = document.getElementById("clear-calibration");
const calibrationStatus = document.getElementById("calibration-status");
const MAX_BASE_OFFSET_SECONDS = 15;
const MAX_DRIFT_SECONDS = 20;

let jobs = [];
let selectedJobId = null;
let availableVoices = [];
let transcriptData = null;
let activeWordIndex = -1;
let activeSectionIndex = -1;
let loadedTranscriptJobId = null;
let transcriptOffsetSeconds = 0;
let transcriptDriftSeconds = 0;
let pendingCalibrationPoint = null;
let calibrationPoints = {
  start: null,
  end: null,
};
let systemStatus = null;

function transcriptSettingsKey(jobId) {
  return `tts-sync-settings:${jobId}`;
}

function calibrationSettingsKey(jobId) {
  return `tts-sync-calibration:${jobId}`;
}

function formatTime(totalSeconds) {
  if (!Number.isFinite(totalSeconds)) {
    return "0:00";
  }
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(seconds / 60);
  const remaining = String(seconds % 60).padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function formatOffset(seconds) {
  if (seconds > 0) {
    return `Earlier ${seconds.toFixed(2)}s`;
  }
  if (seconds < 0) {
    return `Later ${Math.abs(seconds).toFixed(2)}s`;
  }
  return "Aligned";
}

function formatDrift(seconds) {
  if (seconds > 0) {
    return `Earlier ${seconds.toFixed(2)}s by end`;
  }
  if (seconds < 0) {
    return `Later ${Math.abs(seconds).toFixed(2)}s by end`;
  }
  return "No drift";
}

function saveSyncSettings() {
  if (!selectedJobId) {
    return;
  }
  window.localStorage.setItem(
    transcriptSettingsKey(selectedJobId),
    JSON.stringify({
      offset: transcriptOffsetSeconds,
      drift: transcriptDriftSeconds,
    }),
  );
}

function saveCalibrationPoints() {
  if (!selectedJobId) {
    return;
  }
  window.localStorage.setItem(
    calibrationSettingsKey(selectedJobId),
    JSON.stringify(calibrationPoints),
  );
}

function setTranscriptOffset(value, { persist = true, refresh = true } = {}) {
  transcriptOffsetSeconds = Number(value);
  syncOffsetInput.value = String(transcriptOffsetSeconds);
  syncOffsetValue.textContent = formatOffset(transcriptOffsetSeconds);
  if (persist) {
    saveSyncSettings();
  }
  if (refresh) {
    updateTranscriptForCurrentTime(audioElement.currentTime);
  }
}

function setTranscriptDrift(value, { persist = true, refresh = true } = {}) {
  transcriptDriftSeconds = Number(value);
  syncDriftInput.value = String(transcriptDriftSeconds);
  syncDriftValue.textContent = formatDrift(transcriptDriftSeconds);
  if (persist) {
    saveSyncSettings();
  }
  if (refresh) {
    updateTranscriptForCurrentTime(audioElement.currentTime);
  }
}

function loadSavedSyncSettings(jobId) {
  const saved = window.localStorage.getItem(transcriptSettingsKey(jobId));
  if (saved === null) {
    return { offset: 0, drift: 0 };
  }
  try {
    const parsed = JSON.parse(saved);
    return {
      offset: Number.isFinite(Number(parsed.offset)) ? Number(parsed.offset) : 0,
      drift: Number.isFinite(Number(parsed.drift)) ? Number(parsed.drift) : 0,
    };
  } catch {
    return { offset: 0, drift: 0 };
  }
}

function loadSavedCalibrationPoints(jobId) {
  const saved = window.localStorage.getItem(calibrationSettingsKey(jobId));
  if (saved === null) {
    return { start: null, end: null };
  }
  try {
    const parsed = JSON.parse(saved);
    return {
      start: parsed.start ?? null,
      end: parsed.end ?? null,
    };
  } catch {
    return { start: null, end: null };
  }
}

function applySyncSettings(settings, { refresh = true } = {}) {
  setTranscriptOffset(settings.offset, { persist: false, refresh: false });
  setTranscriptDrift(settings.drift, { persist: false, refresh: false });
  if (refresh) {
    updateTranscriptForCurrentTime(audioElement.currentTime);
  }
}

function updateCalibrationStatus() {
  if (pendingCalibrationPoint === "start") {
    calibrationStatus.textContent = "Start point armed. Click the matching transcript word for the current audio position.";
    return;
  }
  if (pendingCalibrationPoint === "end") {
    calibrationStatus.textContent = "End point armed. Click the matching transcript word for the current audio position.";
    return;
  }
  if (calibrationPoints.start && calibrationPoints.end) {
    calibrationStatus.textContent = "Two-point calibration saved for this title.";
    return;
  }
  if (calibrationPoints.start) {
    calibrationStatus.textContent = "Start point saved. Move later in the title, press Set End Point, then click the matching word.";
    return;
  }
  calibrationStatus.textContent = "Two-point calibration: set a start point, then an end point, then click the matching transcript words.";
}

function applyTwoPointCalibration() {
  if (!calibrationPoints.start || !calibrationPoints.end) {
    return false;
  }
  const a1 = Number(calibrationPoints.start.audioTime);
  const t1 = Number(calibrationPoints.start.transcriptTime);
  const a2 = Number(calibrationPoints.end.audioTime);
  const t2 = Number(calibrationPoints.end.transcriptTime);
  const duration = audioElement.duration || transcriptData?.words?.[transcriptData.words.length - 1]?.end_time || 0;
  if (!(duration > 0) || !(a2 > a1)) {
    return false;
  }

  const progress1 = Math.max(0, Math.min(1, a1 / duration));
  const progress2 = Math.max(0, Math.min(1, a2 / duration));
  const delta1 = t1 - a1;
  const delta2 = t2 - a2;
  const progressDiff = progress2 - progress1;

  let nextOffset = delta1;
  let nextDrift = transcriptDriftSeconds;

  if (Math.abs(progressDiff) > 0.001) {
    nextDrift = (delta2 - delta1) / progressDiff;
    nextOffset = delta1 - (nextDrift * progress1);
  }

  nextOffset = Math.max(-MAX_BASE_OFFSET_SECONDS, Math.min(MAX_BASE_OFFSET_SECONDS, nextOffset));
  nextDrift = Math.max(-MAX_DRIFT_SECONDS, Math.min(MAX_DRIFT_SECONDS, nextDrift));

  applySyncSettings({ offset: nextOffset, drift: nextDrift });
  saveCalibrationPoints();
  updateCalibrationStatus();
  return true;
}

function updateSelectedFileLabel() {
  const file = fileInput.files[0];
  selectedFile.textContent = file ? file.name : "No file selected";
  titleInput.disabled = Boolean(file);
}

function renderVoices(voiceItems) {
  availableVoices = voiceItems;
  voiceSelect.innerHTML = "";
  voiceItems.forEach((voice) => {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = `${voice.label} • ${voice.provider === "edge" ? "Premium" : voice.compute_label || "Offline"}`;
    voiceSelect.appendChild(option);
  });
  updateVoiceDescription();
}

function updateVoiceDescription() {
  const current = availableVoices.find((voice) => voice.id === voiceSelect.value) || availableVoices[0];
  if (!current) {
    return;
  }
  const providerLabel = current.provider === "edge"
    ? "Premium online neural voice"
    : `Offline local voice on ${current.compute_label || current.compute_target || "CPU"}`;
  voiceDescription.textContent = `${providerLabel} • ${current.description}${current.compute_note ? ` • ${current.compute_note}` : ""}`;
}

function normalizeSourceType(sourceType, originalFilename = "") {
  if (sourceType === "file" && originalFilename) {
    const suffix = originalFilename.split(".").pop()?.toLowerCase();
    if (["txt", "md", "csv", "log"].includes(suffix)) {
      return "txt";
    }
    if (suffix === "pdf" || suffix === "epub") {
      return suffix;
    }
  }
  if (["txt", "md", "csv", "log"].includes(sourceType)) {
    return "txt";
  }
  return sourceType;
}

function sourceTypeLabel(job) {
  const normalized = normalizeSourceType(job.source_type, job.original_filename);
  if (normalized === "paste") {
    return "pasted";
  }
  if (normalized === "txt") {
    return "text";
  }
  return normalized;
}

function formatRelativeTime(timestamp) {
  if (!timestamp) {
    return "Not played yet";
  }
  const secondsAgo = Math.max(0, Math.round((Date.now() / 1000) - timestamp));
  if (secondsAgo < 60) {
    return "Listened just now";
  }
  if (secondsAgo < 3600) {
    return `Listened ${Math.floor(secondsAgo / 60)}m ago`;
  }
  if (secondsAgo < 86400) {
    return `Listened ${Math.floor(secondsAgo / 3600)}h ago`;
  }
  return `Listened ${Math.floor(secondsAgo / 86400)}d ago`;
}

function formatDurationCompact(totalSeconds) {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) {
    return "0s";
  }
  const rounded = Math.round(totalSeconds);
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;
  if (minutes <= 0) {
    return `${seconds}s`;
  }
  if (minutes < 60) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

function formatUpdatedTime(timestamp) {
  if (!timestamp) {
    return "Not loaded";
  }
  const ageSeconds = Math.max(0, Math.round((Date.now() / 1000) - timestamp));
  if (ageSeconds <= 2) {
    return "Just updated";
  }
  return `${ageSeconds}s ago`;
}

function processingMeta(job) {
  const elapsed = Number(job.processing_elapsed_seconds);
  const progress = Number(job.progress || 0);
  if (job.state === "processing" && Number.isFinite(elapsed) && elapsed > 0) {
    if (progress > 0.03) {
      const estimatedTotal = elapsed / progress;
      const remaining = Math.max(0, estimatedTotal - elapsed);
      return `Preparing for ${formatDurationCompact(elapsed)} • ETA ${formatDurationCompact(remaining)}`;
    }
    return `Preparing for ${formatDurationCompact(elapsed)}`;
  }
  if (job.state === "completed" && Number.isFinite(Number(job.completed_in_seconds))) {
    return `Ready in ${formatDurationCompact(Number(job.completed_in_seconds))}`;
  }
  if (job.state === "failed" && Number.isFinite(elapsed) && elapsed > 0) {
    return `Stopped after ${formatDurationCompact(elapsed)}`;
  }
  return null;
}

function activeQueueJobs() {
  return [...jobs]
    .filter((job) => job.state === "processing" || job.state === "queued")
    .sort((left, right) => {
      const stateWeight = (job) => (job.state === "processing" ? 0 : 1);
      return stateWeight(left) - stateWeight(right) || left.created_at - right.created_at;
    });
}

function renderActiveQueue() {
  const activeJobs = activeQueueJobs();
  queuePanelCount.textContent = `${activeJobs.length} active`;
  queuePanelTitle.textContent = activeJobs.length
    ? `${activeJobs.filter((job) => job.state === "processing").length || 0} preparing • ${activeJobs.filter((job) => job.state === "queued").length || 0} queued`
    : "Nothing is being prepared right now";

  if (!activeJobs.length) {
    activeJobList.innerHTML = `<div class="empty-state">Items you add will appear here while Rayline Echo prepares them for listening.</div>`;
    return;
  }

  activeJobList.innerHTML = "";
  activeJobs.forEach((job, index) => {
    const progressPercent = Math.max(3, Math.round(job.progress * 100));
    const item = document.createElement("article");
    item.className = `active-job-card ${job.state === "processing" ? "is-processing" : "is-queued"}`;
    item.innerHTML = `
      <div class="active-job-topline">
        <strong>${job.title}</strong>
        <span class="status-pill status-${job.state}">${job.state === "processing" ? "Preparing" : `Queued #${index + 1 - activeJobs.filter((entry) => entry.state === "processing").length}`}</span>
      </div>
      <p class="job-meta">${sourceTypeLabel(job)} • ${job.voice_label} • ${job.compute_target === "cloud" ? "cloud" : `local ${job.compute_target || "cpu"}`}</p>
      <div class="job-progress" aria-hidden="true">
        <span style="width:${job.state === "processing" ? progressPercent : 8}%"></span>
      </div>
      <p class="job-meta">${processingMeta(job) || (job.state === "queued" ? "Waiting in queue" : statusLabel(job))}</p>
    `;
    activeJobList.appendChild(item);
  });
}

function renderSystemStatus() {
  const cpu = systemStatus?.cpu || { available: false, summary: "CPU stats unavailable right now." };
  const gpu = systemStatus?.gpu || { available: false, summary: "GPU stats unavailable right now." };
  const piper = systemStatus?.piper || { label: "CPU", using_cuda: false };

  const cpuBusy = Number(cpu.cpu_busy_percent || 0);
  const gpuBusy = Number(gpu.gpu_util_percent || 0);

  systemPanelTitle.textContent = activeQueueJobs().length
    ? `Machine load while your queue is running • Local voices on ${piper.label}`
    : `Machine load while the app is idle • Local voices on ${piper.label}`;
  systemPanelUpdated.textContent = formatUpdatedTime(systemStatus?.timestamp);

  cpuBusyPill.textContent = cpu.available ? `${cpuBusy}%` : "--";
  cpuSummary.textContent = cpu.summary || "CPU stats unavailable right now.";
  cpuMeterFill.style.width = `${Math.max(0, Math.min(100, cpuBusy))}%`;
  cpuBreakdown.textContent = cpu.available
    ? `User ${cpu.user_percent}% • System ${cpu.system_percent}% • Wait ${cpu.wait_percent}%`
    : "User -- • System -- • Wait --";

  gpuBusyPill.textContent = gpu.available ? `${gpuBusy}%` : "--";
  gpuSummary.textContent = gpu.summary || "GPU stats unavailable right now.";
  gpuMeterFill.style.width = `${Math.max(0, Math.min(100, gpuBusy))}%`;
  gpuBreakdown.textContent = gpu.available
    ? `VRAM ${gpu.memory_used_mb}/${gpu.memory_total_mb} MB • Temp ${gpu.temperature_c}C`
    : "VRAM -- • Temp --";
}

function filteredJobs() {
  const filter = libraryFilterSelect.value;
  const query = librarySearchInput.value.trim().toLowerCase();
  const sort = librarySortSelect.value;
  let visible = [...jobs];

  if (filter === "favorites") {
    visible = visible.filter((job) => job.favorite);
  } else if (filter === "recent") {
    visible = visible.filter((job) => job.is_recent);
  } else if (filter !== "all") {
    visible = visible.filter((job) => normalizeSourceType(job.source_type, job.original_filename) === filter);
  }

  if (query) {
    visible = visible.filter((job) => {
      const haystack = [
        job.title,
        job.original_filename || "",
        job.preview || "",
        job.voice_label || "",
        sourceTypeLabel(job),
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    });
  }

  visible.sort((left, right) => {
    if (sort === "newest") {
      return right.created_at - left.created_at;
    }
    if (sort === "oldest") {
      return left.created_at - right.created_at;
    }
    if (sort === "title") {
      return left.title.localeCompare(right.title);
    }
    if (sort === "length") {
      return (right.duration_seconds || 0) - (left.duration_seconds || 0);
    }
    return (
      Number(right.favorite) - Number(left.favorite)
      || (right.last_played_at || 0) - (left.last_played_at || 0)
      || right.created_at - left.created_at
    );
  });

  return visible;
}

function statusLabel(job) {
  if (job.state === "processing") {
    return `Preparing ${Math.round(job.progress * 100)}%`;
  }
  if (job.state === "completed") {
    return "Ready";
  }
  if (job.state === "failed") {
    return "Failed";
  }
  return "Queued";
}

function renderJobs() {
  const visibleJobs = filteredJobs();
  jobCount.textContent = `${visibleJobs.length} ${visibleJobs.length === 1 ? "title" : "titles"}`;
  if (!jobs.length) {
    jobList.innerHTML = `<div class="empty-state">Your library is empty. Add a document, book, or note to create your first audiobook.</div>`;
    return;
  }
  if (!visibleJobs.length) {
    jobList.innerHTML = `<div class="empty-state">No saved titles match this view yet.</div>`;
    return;
  }

  jobList.innerHTML = "";
  visibleJobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = `job-card ${job.id === selectedJobId ? "is-selected" : ""}`;
    const progressPercent = Math.max(3, Math.round(job.progress * 100));
    const canPlay = job.state === "completed";
    const showResume = job.state === "failed" && job.resumable;
    card.innerHTML = `
      <div class="job-topline">
        <div class="job-heading">
          <div class="job-title-row">
            <h3 class="job-title">${job.title}</h3>
            ${job.favorite ? '<span class="job-badge favorite-badge">Favorite</span>' : ""}
            ${job.is_recent ? '<span class="job-badge recent-badge">Recent</span>' : ""}
          </div>
          <p class="job-meta">${sourceTypeLabel(job)} • ${job.voice_label} • ${job.compute_target === "cloud" ? "cloud" : `local ${job.compute_target || "cpu"}`} • ${job.text_length.toLocaleString()} characters</p>
        </div>
        <span class="status-pill status-${job.state}">${statusLabel(job)}</span>
      </div>
      <div class="job-progress" aria-hidden="true">
        <span style="width:${progressPercent}%"></span>
      </div>
      <p class="job-meta">Sections ${job.completed_chunks}/${job.total_chunks || "?"} • ${formatRelativeTime(job.last_played_at)} • ${job.duration_seconds ? `${formatTime(job.duration_seconds)} long` : "Preparing audio"}</p>
      ${processingMeta(job) ? `<p class="job-timing">${processingMeta(job)}</p>` : ""}
      <p class="job-preview">${job.error || job.preview}</p>
      <div class="job-management">
        <div class="job-primary-actions">
          <button type="button" class="library-button ${showResume ? "action-resume" : "action-listen"} ${job.id === selectedJobId ? "is-selected-action" : ""}" data-action="${showResume ? "resume" : "listen"}" data-job-id="${job.id}" ${canPlay || showResume ? "" : "disabled"}>${showResume ? "Resume" : job.id === selectedJobId ? "Current" : "Listen"}</button>
          <button type="button" class="library-button action-favorite" data-action="favorite" data-job-id="${job.id}">${job.favorite ? "Unfavorite" : "Favorite"}</button>
          <button type="button" class="library-button action-rename" data-action="rename" data-job-id="${job.id}">Rename</button>
        </div>
        <div class="job-actions job-actions-secondary">
          <label class="inline-select voice-inline-select">
            <span>Create another version</span>
            <select data-job-id="${job.id}" class="reprocess-voice-select">
              ${availableVoices.map((voice) => `<option value="${voice.id}" ${voice.id === job.voice ? "selected" : ""}>${voice.label}</option>`).join("")}
            </select>
          </label>
          <div class="job-secondary-actions">
            <button type="button" class="library-button action-reprocess" data-action="reprocess" data-job-id="${job.id}">Create version</button>
            <button type="button" class="library-button danger-button action-delete" data-action="delete" data-job-id="${job.id}">Delete</button>
          </div>
        </div>
      </div>
    `;
    jobList.appendChild(card);
  });
}

function renderTranscript(data) {
  transcriptData = data;
  activeWordIndex = -1;
  activeSectionIndex = -1;
  transcriptViewer.innerHTML = "";
  renderSectionNav(data);

  if (!data || !data.text || !data.words || !data.words.length) {
    transcriptViewer.innerHTML = `<p class="transcript-empty">Read-along timing is not available for this title.</p>`;
    readerMode.textContent = "Read-along timing is not available";
    return;
  }

  readerMode.textContent = data.timing_mode === "exact"
    ? "Word timing from the speech engine"
    : "Estimated word timing for offline voice";

  let cursor = 0;
  data.words.forEach((word, index) => {
    if (cursor < word.char_start) {
      transcriptViewer.append(document.createTextNode(data.text.slice(cursor, word.char_start)));
    }
    const span = document.createElement("span");
    span.className = "transcript-word";
    span.dataset.wordIndex = String(index);
    span.textContent = data.text.slice(word.char_start, word.char_end);
    if (typeof word.start_time === "number") {
      span.dataset.startTime = String(word.start_time);
    }
    transcriptViewer.append(span);
    cursor = word.char_end;
  });

  if (cursor < data.text.length) {
    transcriptViewer.append(document.createTextNode(data.text.slice(cursor)));
  }
}

function renderSectionNav(data) {
  sectionNav.innerHTML = "";
  const sections = Array.isArray(data?.sections) ? data.sections.filter((section) => section?.title) : [];

  if (!sections.length) {
    sectionCount.textContent = "No sections yet";
    sectionNav.innerHTML = `<p class="transcript-empty">Chapter or section markers will appear here when available.</p>`;
    return;
  }

  sectionCount.textContent = sections.length === 1 ? "1 section" : `${sections.length} sections`;
  sections.forEach((section, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "section-chip";
    button.dataset.sectionIndex = String(index);
    button.textContent = `${index + 1}. ${section.title}`;
    sectionNav.append(button);
  });
}

function setActiveSection(index) {
  if (index === activeSectionIndex) {
    return;
  }

  const previous = sectionNav.querySelector(".section-chip.is-active");
  if (previous) {
    previous.classList.remove("is-active");
  }

  activeSectionIndex = index;
  if (index < 0) {
    return;
  }

  const next = sectionNav.querySelector(`[data-section-index="${index}"]`);
  if (!next) {
    return;
  }

  next.classList.add("is-active");
  if (!audioElement.paused && followPlaybackInput.checked) {
    next.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }
}

function findActiveSectionIndex(adjustedTime, activeIndex) {
  const sections = transcriptData?.sections;
  if (!Array.isArray(sections) || !sections.length) {
    return -1;
  }

  for (let index = 0; index < sections.length; index += 1) {
    const section = sections[index];
    if (
      Number.isFinite(section.start_time)
      && Number.isFinite(section.end_time)
      && adjustedTime >= section.start_time
      && adjustedTime < section.end_time
    ) {
      return index;
    }
  }

  if (activeIndex >= 0) {
    const activeWord = transcriptData?.words?.[activeIndex];
    if (activeWord) {
      for (let index = 0; index < sections.length; index += 1) {
        const section = sections[index];
        if (
          Number.isFinite(section.char_start)
          && Number.isFinite(section.char_end)
          && activeWord.char_start >= section.char_start
          && activeWord.char_end <= section.char_end
        ) {
          return index;
        }
      }
    }
  }

  return adjustedTime > 0 ? sections.length - 1 : -1;
}

function jumpToSection(index) {
  const section = transcriptData?.sections?.[index];
  if (!section) {
    return;
  }

  if (Number.isFinite(section.start_time) && Number.isFinite(audioElement.duration)) {
    audioElement.currentTime = Math.max(0, Math.min(audioElement.duration, section.start_time));
  } else if (Number.isFinite(section.start_time)) {
    audioElement.currentTime = Math.max(0, section.start_time);
  }

  if (Number.isFinite(section.char_start)) {
    const targetWord = transcriptData?.words?.find((word) => word.char_start >= section.char_start);
    if (targetWord) {
      const wordElement = transcriptViewer.querySelector(`[data-word-index="${transcriptData.words.indexOf(targetWord)}"]`);
      if (wordElement) {
        wordElement.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
      }
    }
  }

  setActiveSection(index);
  updateTranscriptForCurrentTime(audioElement.currentTime);
}

function setActiveWord(index) {
  if (!transcriptData || index === activeWordIndex) {
    return;
  }

  const previous = transcriptViewer.querySelector(".transcript-word.is-active");
  if (previous) {
    previous.classList.remove("is-active");
  }

  activeWordIndex = index;
  if (index < 0) {
    return;
  }

  const next = transcriptViewer.querySelector(`[data-word-index="${index}"]`);
  if (!next) {
    return;
  }
  next.classList.add("is-active");
  if (!audioElement.paused && followPlaybackInput.checked) {
    next.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
  }
}

function updateTranscriptForCurrentTime(currentTime) {
  if (!transcriptData || !transcriptData.words || !transcriptData.words.length) {
    return;
  }
  const duration = audioElement.duration || transcriptData.words[transcriptData.words.length - 1].end_time || 0;
  const progress = duration > 0 ? Math.max(0, Math.min(1, currentTime / duration)) : 0;
  const adjustedTime = Math.max(0, currentTime + transcriptOffsetSeconds + (transcriptDriftSeconds * progress));

  let foundIndex = -1;
  for (let index = 0; index < transcriptData.words.length; index += 1) {
    const word = transcriptData.words[index];
    if (adjustedTime >= word.start_time && adjustedTime < word.end_time) {
      foundIndex = index;
      break;
    }
  }

  if (foundIndex === -1 && adjustedTime >= transcriptData.words[transcriptData.words.length - 1].end_time) {
    foundIndex = transcriptData.words.length - 1;
  }

  setActiveWord(foundIndex);
  setActiveSection(findActiveSectionIndex(adjustedTime, foundIndex));
}

async function loadTranscript(job) {
  if (!job || !job.transcript_url) {
    transcriptData = null;
    activeWordIndex = -1;
    activeSectionIndex = -1;
    loadedTranscriptJobId = null;
    sectionCount.textContent = "No sections yet";
    sectionNav.innerHTML = `<p class="transcript-empty">Chapter or section markers will appear here when available.</p>`;
    transcriptViewer.innerHTML = `<p class="transcript-empty">Choose a completed title to follow along as it plays.</p>`;
    readerMode.textContent = "Waiting for a completed title";
    return;
  }
  if (loadedTranscriptJobId === job.id && transcriptData) {
    return;
  }
  transcriptData = null;
  activeWordIndex = -1;
  const response = await fetch(job.transcript_url);
  if (!response.ok) {
    loadedTranscriptJobId = null;
    activeSectionIndex = -1;
    sectionCount.textContent = "No sections yet";
    sectionNav.innerHTML = `<p class="transcript-empty">Chapter or section markers will appear here when available.</p>`;
    transcriptViewer.innerHTML = `<p class="transcript-empty">Read-along timing is not available for this title.</p>`;
    readerMode.textContent = "Transcript unavailable";
    return;
  }
  const data = await response.json();
  loadedTranscriptJobId = job.id;
  renderTranscript(data);
  applySyncSettings(loadSavedSyncSettings(job.id));
  calibrationPoints = loadSavedCalibrationPoints(job.id);
  pendingCalibrationPoint = null;
  updateCalibrationStatus();
}

function syncSelectedJob() {
  if (!selectedJobId) {
    return;
  }
  const current = jobs.find((job) => job.id === selectedJobId);
  if (!current || current.state !== "completed") {
    return;
  }
  if (audioElement.src !== new URL(current.audio_url, window.location.origin).href) {
    audioElement.src = current.audio_url;
  }
  trackTitle.textContent = current.title;
  trackVoice.textContent = `${current.voice_label} • ${current.provider === "edge" ? "premium neural on cloud" : `offline local on ${current.compute_target || "cpu"}`}`;
  playerSubtitle.textContent = `${current.text_length.toLocaleString()} characters • ${formatTime(current.duration_seconds || 0)}`;
  downloadLink.href = current.download_url;
  downloadLink.textContent = `Download ${String(current.audio_format || "").toUpperCase()}`;
  downloadLink.classList.remove("hidden");
  applySyncSettings(loadSavedSyncSettings(current.id), { refresh: false });
  calibrationPoints = loadSavedCalibrationPoints(current.id);
  pendingCalibrationPoint = null;
  updateCalibrationStatus();
  loadTranscript(current);
}

async function selectJob(jobId) {
  const job = jobs.find((item) => item.id === jobId);
  if (!job || job.state !== "completed") {
    return;
  }
  selectedJobId = jobId;
  audioElement.pause();
  audioElement.currentTime = 0;
  timeline.value = 0;
  currentTimeEl.textContent = "0:00";
  loadedTranscriptJobId = null;
  pendingCalibrationPoint = null;
  syncSelectedJob();
  try {
    await patchJob(jobId, { touch_recent: true });
  } catch {
    // Keep selection usable even if the recents update fails.
  }
  const selected = jobs.find((item) => item.id === jobId);
  if (selected) {
    selected.last_played_at = Date.now() / 1000;
    selected.is_recent = true;
  }
  renderJobs();
}

function nudgeTranscriptOffset(delta) {
  const next = Math.max(-MAX_BASE_OFFSET_SECONDS, Math.min(MAX_BASE_OFFSET_SECONDS, transcriptOffsetSeconds + delta));
  setTranscriptOffset(next);
}

function nudgeTranscriptDrift(delta) {
  const next = Math.max(-MAX_DRIFT_SECONDS, Math.min(MAX_DRIFT_SECONDS, transcriptDriftSeconds + delta));
  setTranscriptDrift(next);
}

async function loadVoices() {
  const response = await fetch("/api/voices");
  const data = await response.json();
  renderVoices(data.voices);
}

async function loadJobs() {
  const response = await fetch("/api/jobs");
  const data = await response.json();
  jobs = data.jobs;
  if (data.system) {
    systemStatus = data.system;
  }

  if (!selectedJobId) {
    const newestCompleted = jobs.find((job) => job.state === "completed");
    if (newestCompleted) {
      selectedJobId = newestCompleted.id;
    }
  }

  syncSelectedJob();
  renderJobs();
  renderActiveQueue();
  renderSystemStatus();
}

async function loadSystemStatus() {
  try {
    const response = await fetch("/api/system");
    if (!response.ok) {
      throw new Error("Unable to load system metrics.");
    }
    systemStatus = await response.json();
  } catch {
    systemStatus = {
      timestamp: Date.now() / 1000,
      cpu: { available: false, summary: "CPU stats unavailable right now." },
      gpu: { available: false, summary: "GPU stats unavailable right now." },
    };
  }
  renderSystemStatus();
}

async function refreshJobsKeepingSelection() {
  await loadJobs();
  if (selectedJobId && !jobs.some((job) => job.id === selectedJobId)) {
    selectedJobId = null;
    audioElement.pause();
    audioElement.removeAttribute("src");
    audioElement.load();
    downloadLink.classList.add("hidden");
    trackTitle.textContent = "Nothing selected";
    trackVoice.textContent = "No voice selected";
    playerSubtitle.textContent = "Choose something from your library to start listening.";
    readerMode.textContent = "Waiting for a completed title";
    transcriptViewer.innerHTML = `<p class="transcript-empty">Choose a completed title to follow along as it plays.</p>`;
    sectionCount.textContent = "No sections yet";
    sectionNav.innerHTML = `<p class="transcript-empty">Chapter or section markers will appear here when available.</p>`;
  }
}

async function patchJob(jobId, payload) {
  const response = await fetch(`/api/jobs/${jobId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Unable to update this title.");
  }
  return response.json();
}

async function deleteJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Unable to delete this title.");
  }
  return response.json();
}

async function reprocessJob(jobId, voice) {
  const response = await fetch(`/api/jobs/${jobId}/reprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ voice }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Unable to create another version.");
  }
  return response.json();
}

async function resumeJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/resume`, {
    method: "POST",
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Unable to resume this title.");
  }
  return response.json();
}

async function submitJob(event) {
  event.preventDefault();

  const payload = new FormData();
  payload.append("voice", voiceSelect.value);
  payload.append("title", titleInput.value);
  payload.append("text", textInput.value);
  if (fileInput.files[0]) {
    payload.append("file", fileInput.files[0]);
  }

  const response = await fetch("/api/jobs", {
    method: "POST",
    body: payload,
  });

  if (!response.ok) {
    const error = await response.json();
    alert(error.detail || "Unable to add this item to the queue.");
    return;
  }

  titleInput.value = "";
  textInput.value = "";
  fileInput.value = "";
  updateSelectedFileLabel();
  await loadJobs();
}

async function renameTrack(jobId) {
  const current = jobs.find((job) => job.id === jobId);
  if (!current) {
    return;
  }
  const nextTitle = window.prompt("Rename this title:", current.title);
  if (nextTitle === null) {
    return;
  }
  if (!nextTitle.trim()) {
    window.alert("Title cannot be empty.");
    return;
  }
  await patchJob(jobId, { title: nextTitle.trim() });
  await refreshJobsKeepingSelection();
  if (selectedJobId === jobId) {
    syncSelectedJob();
  }
}

async function toggleFavorite(jobId) {
  const current = jobs.find((job) => job.id === jobId);
  if (!current) {
    return;
  }
  await patchJob(jobId, { favorite: !current.favorite });
  await refreshJobsKeepingSelection();
}

async function removeTrack(jobId) {
  const current = jobs.find((job) => job.id === jobId);
  if (!current) {
    return;
  }
  const confirmed = window.confirm(`Delete "${current.title}" and its generated audio?`);
  if (!confirmed) {
    return;
  }
  await deleteJob(jobId);
  if (selectedJobId === jobId) {
    selectedJobId = null;
  }
  await refreshJobsKeepingSelection();
}

async function queueReprocess(jobId) {
  const select = jobList.querySelector(`.reprocess-voice-select[data-job-id="${jobId}"]`);
  const voice = select?.value;
  if (!voice) {
    window.alert("Choose a voice first.");
    return;
  }
  await reprocessJob(jobId, voice);
  await refreshJobsKeepingSelection();
}

async function resumeTrackedJob(jobId) {
  await resumeJob(jobId);
  await refreshJobsKeepingSelection();
}

function clearForm() {
  titleInput.value = "";
  textInput.value = "";
  fileInput.value = "";
  updateSelectedFileLabel();
}

function handleDrop(event) {
  event.preventDefault();
  dropzone.classList.remove("drag-active");
  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  fileInput.files = dataTransfer.files;
  updateSelectedFileLabel();
}

function openFilePicker() {
  fileInput.click();
}

playButton.addEventListener("click", async () => {
  if (!audioElement.src) {
    return;
  }
  await audioElement.play();
});

pauseButton.addEventListener("click", () => {
  audioElement.pause();
});

backButton.addEventListener("click", () => {
  audioElement.currentTime = Math.max(0, audioElement.currentTime - 5);
});

forwardButton.addEventListener("click", () => {
  audioElement.currentTime = Math.min(audioElement.duration || 0, audioElement.currentTime + 5);
});

timeline.addEventListener("input", () => {
  if (!audioElement.duration) {
    return;
  }
  audioElement.currentTime = (Number(timeline.value) / 100) * audioElement.duration;
});

audioElement.addEventListener("timeupdate", () => {
  currentTimeEl.textContent = formatTime(audioElement.currentTime);
  durationTimeEl.textContent = formatTime(audioElement.duration);
  if (audioElement.duration) {
    timeline.value = ((audioElement.currentTime / audioElement.duration) * 100).toFixed(1);
  } else {
    timeline.value = 0;
  }
  updateTranscriptForCurrentTime(audioElement.currentTime);
});

audioElement.addEventListener("loadedmetadata", () => {
  durationTimeEl.textContent = formatTime(audioElement.duration);
  updateTranscriptForCurrentTime(audioElement.currentTime);
});

audioElement.addEventListener("seeked", () => {
  updateTranscriptForCurrentTime(audioElement.currentTime);
});

transcriptViewer.addEventListener("click", (event) => {
  const target = event.target.closest(".transcript-word");
  if (!target) {
    return;
  }
  const startTime = Number(target.dataset.startTime);
  if (Number.isFinite(startTime)) {
    if (pendingCalibrationPoint) {
      calibrationPoints[pendingCalibrationPoint] = {
        audioTime: audioElement.currentTime,
        transcriptTime: startTime,
      };
      pendingCalibrationPoint = null;
      if (!applyTwoPointCalibration()) {
        saveCalibrationPoints();
        updateCalibrationStatus();
      }
      return;
    }
    const duration = audioElement.duration || transcriptData?.words?.[transcriptData.words.length - 1]?.end_time || 0;
    const progress = duration > 0 ? Math.max(0, Math.min(1, audioElement.currentTime / duration)) : 0;
    const currentAdjusted = audioElement.currentTime + transcriptOffsetSeconds + (transcriptDriftSeconds * progress);
    const delta = startTime - currentAdjusted;

    if (progress > 0.2) {
      const nextDrift = Math.max(-MAX_DRIFT_SECONDS, Math.min(MAX_DRIFT_SECONDS, transcriptDriftSeconds + (delta / progress)));
      setTranscriptDrift(nextDrift, { refresh: false });
    } else {
      const nextOffset = Math.max(-MAX_BASE_OFFSET_SECONDS, Math.min(MAX_BASE_OFFSET_SECONDS, transcriptOffsetSeconds + delta));
      setTranscriptOffset(nextOffset, { refresh: false });
    }
    updateTranscriptForCurrentTime(audioElement.currentTime);
  }
});

sectionNav.addEventListener("click", (event) => {
  const target = event.target.closest(".section-chip");
  if (!target) {
    return;
  }
  jumpToSection(Number(target.dataset.sectionIndex));
});

jobList.addEventListener("click", async (event) => {
  const target = event.target.closest("button[data-action]");
  if (!target) {
    return;
  }

  const { action, jobId } = target.dataset;
  if (!jobId || !action) {
    return;
  }

  try {
    if (action === "listen") {
      await selectJob(jobId);
      return;
    }
    if (action === "favorite") {
      await toggleFavorite(jobId);
      return;
    }
    if (action === "resume") {
      await resumeTrackedJob(jobId);
      return;
    }
    if (action === "rename") {
      await renameTrack(jobId);
      return;
    }
    if (action === "delete") {
      await removeTrack(jobId);
      return;
    }
    if (action === "reprocess") {
      await queueReprocess(jobId);
    }
  } catch (error) {
    window.alert(error.message || "Something went wrong while updating your library.");
  }
});

fileInput.addEventListener("change", updateSelectedFileLabel);
voiceSelect.addEventListener("change", updateVoiceDescription);
librarySearchInput.addEventListener("input", renderJobs);
libraryFilterSelect.addEventListener("change", renderJobs);
librarySortSelect.addEventListener("change", renderJobs);
form.addEventListener("submit", submitJob);
clearButton.addEventListener("click", clearForm);
syncOffsetInput.addEventListener("input", () => {
  setTranscriptOffset(Number(syncOffsetInput.value));
});
syncDriftInput.addEventListener("input", () => {
  setTranscriptDrift(Number(syncDriftInput.value));
});
syncEarlierButton.addEventListener("click", () => {
  nudgeTranscriptOffset(0.25);
});
syncLaterButton.addEventListener("click", () => {
  nudgeTranscriptOffset(-0.25);
});
driftLessButton.addEventListener("click", () => {
  nudgeTranscriptDrift(-0.5);
});
driftMoreButton.addEventListener("click", () => {
  nudgeTranscriptDrift(0.5);
});
syncResetButton.addEventListener("click", () => {
  applySyncSettings({ offset: 0, drift: 0 });
});
calibrateStartButton.addEventListener("click", () => {
  pendingCalibrationPoint = "start";
  updateCalibrationStatus();
});
calibrateEndButton.addEventListener("click", () => {
  pendingCalibrationPoint = "end";
  updateCalibrationStatus();
});
clearCalibrationButton.addEventListener("click", () => {
  pendingCalibrationPoint = null;
  calibrationPoints = { start: null, end: null };
  if (selectedJobId) {
    window.localStorage.removeItem(calibrationSettingsKey(selectedJobId));
  }
  updateCalibrationStatus();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "dragleave") {
      dropzone.classList.remove("drag-active");
    }
  });
});

dropzone.addEventListener("drop", handleDrop);
dropzone.addEventListener("click", (event) => {
  if (event.target === fileInput) {
    return;
  }
  event.preventDefault();
  openFilePicker();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  event.preventDefault();
  openFilePicker();
});

async function init() {
  renderSystemStatus();
  await loadVoices();
  updateSelectedFileLabel();
  await loadJobs();
  await loadSystemStatus();
  setInterval(loadJobs, 1500);
  setInterval(loadSystemStatus, 3000);
}

init();
