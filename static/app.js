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
const MAX_BASE_OFFSET_SECONDS = 3;
const MAX_DRIFT_SECONDS = 20;

let jobs = [];
let selectedJobId = null;
let availableVoices = [];
let transcriptData = null;
let activeWordIndex = -1;
let loadedTranscriptJobId = null;
let transcriptOffsetSeconds = 0;
let transcriptDriftSeconds = 0;
let pendingCalibrationPoint = null;
let calibrationPoints = {
  start: null,
  end: null,
};

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
    calibrationStatus.textContent = "Two-point calibration saved for this track.";
    return;
  }
  if (calibrationPoints.start) {
    calibrationStatus.textContent = "Start point saved. Move later in the track, press Set End Point, then click the matching word.";
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
    option.textContent = `${voice.label} • ${voice.provider === "edge" ? "Premium" : "Offline"}`;
    voiceSelect.appendChild(option);
  });
  updateVoiceDescription();
}

function updateVoiceDescription() {
  const current = availableVoices.find((voice) => voice.id === voiceSelect.value) || availableVoices[0];
  if (!current) {
    return;
  }
  const providerLabel = current.provider === "edge" ? "Premium online neural voice" : "Offline local voice";
  voiceDescription.textContent = `${providerLabel} • ${current.description}`;
}

function statusLabel(job) {
  if (job.state === "processing") {
    return `Processing ${Math.round(job.progress * 100)}%`;
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
  jobCount.textContent = `${jobs.length} ${jobs.length === 1 ? "job" : "jobs"}`;
  if (!jobs.length) {
    jobList.innerHTML = `<div class="empty-state">No tracks yet. Submit text to start building your library.</div>`;
    return;
  }

  jobList.innerHTML = "";
  jobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = `job-card ${job.id === selectedJobId ? "is-selected" : ""}`;
    const progressPercent = Math.max(3, Math.round(job.progress * 100));
    const canPlay = job.state === "completed";
    card.innerHTML = `
      <div class="job-topline">
        <div>
          <h3 class="job-title">${job.title}</h3>
          <p class="job-meta">${job.source_type} • ${job.voice_label} • ${job.text_length.toLocaleString()} characters</p>
        </div>
        <span class="status-pill status-${job.state}">${statusLabel(job)}</span>
      </div>
      <div class="job-progress" aria-hidden="true">
        <span style="width:${progressPercent}%"></span>
      </div>
      <p class="job-meta">Chunks ${job.completed_chunks}/${job.total_chunks || "?"}</p>
      <p class="job-preview">${job.error || job.preview}</p>
      <div class="job-actions">
        <span class="job-meta">${job.duration_seconds ? `${formatTime(job.duration_seconds)} long` : "Waiting for audio"}</span>
        <button type="button" data-job-id="${job.id}" ${canPlay ? "" : "disabled"}>${job.id === selectedJobId ? "Selected" : "Listen"}</button>
      </div>
    `;
    const button = card.querySelector("button");
    if (button) {
      button.addEventListener("click", () => selectJob(job.id));
    }
    jobList.appendChild(card);
  });
}

function renderTranscript(data) {
  transcriptData = data;
  activeWordIndex = -1;
  transcriptViewer.innerHTML = "";

  if (!data || !data.text || !data.words || !data.words.length) {
    transcriptViewer.innerHTML = `<p class="transcript-empty">Transcript timing is not available for this track.</p>`;
    readerMode.textContent = "No transcript timing available";
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
}

async function loadTranscript(job) {
  if (!job || !job.transcript_url) {
    transcriptData = null;
    activeWordIndex = -1;
    loadedTranscriptJobId = null;
    transcriptViewer.innerHTML = `<p class="transcript-empty">Pick a finished track to follow the spoken text word by word.</p>`;
    readerMode.textContent = "Waiting for a completed track";
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
    transcriptViewer.innerHTML = `<p class="transcript-empty">Transcript timing is not available for this track.</p>`;
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
  trackVoice.textContent = `${current.voice_label} • ${current.provider === "edge" ? "premium neural" : "offline local"}`;
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

function selectJob(jobId) {
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

  if (!selectedJobId) {
    const newestCompleted = jobs.find((job) => job.state === "completed");
    if (newestCompleted) {
      selectedJobId = newestCompleted.id;
    }
  }

  syncSelectedJob();
  renderJobs();
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
    alert(error.detail || "Unable to create TTS job.");
    return;
  }

  titleInput.value = "";
  textInput.value = "";
  fileInput.value = "";
  updateSelectedFileLabel();
  await loadJobs();
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

fileInput.addEventListener("change", updateSelectedFileLabel);
voiceSelect.addEventListener("change", updateVoiceDescription);
form.addEventListener("submit", submitJob);
clearButton.addEventListener("click", clearForm);
syncOffsetInput.addEventListener("input", () => {
  setTranscriptOffset(Number(syncOffsetInput.value));
});
syncDriftInput.addEventListener("input", () => {
  setTranscriptDrift(Number(syncDriftInput.value));
});
syncEarlierButton.addEventListener("click", () => {
  nudgeTranscriptOffset(0.1);
});
syncLaterButton.addEventListener("click", () => {
  nudgeTranscriptOffset(-0.1);
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

async function init() {
  await loadVoices();
  updateSelectedFileLabel();
  await loadJobs();
  setInterval(loadJobs, 1500);
}

init();
