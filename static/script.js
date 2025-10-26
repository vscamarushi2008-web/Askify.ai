// Simple frontend logic: drag/drop, upload, ask question, show history
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const dropzone = document.getElementById('dropzone');
const fileList = document.getElementById('fileList');

const askBtn = document.getElementById('askBtn');
const questionInput = document.getElementById('questionInput');
const processing = document.getElementById('processing');
const answerBox = document.getElementById('answerBox');
const answerText = document.getElementById('answerText');

const viewHistoryBtn = document.getElementById('viewHistoryBtn');
const historyModal = document.getElementById('historyModal');
const historyList = document.getElementById('historyList');
const closeHistory = document.getElementById('closeHistory');
const clearHistory = document.getElementById('clearHistory');
const closeBtn = document.getElementById('closeBtn');

let uploadedFiles = [];
let history = [];

// upload handling (client -> server)
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    await uploadFiles(files);
});

// drag/drop visuals (optional)
['dragenter','dragover'].forEach(ev => {
    dropzone.addEventListener(ev, (e)=>{
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.add('drag-over');
    });
});
['dragleave','drop'].forEach(ev => {
    dropzone.addEventListener(ev, (e)=>{
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.remove('drag-over');
    });
});

dropzone.addEventListener('drop', async (e) => {
    const dt = e.dataTransfer;
    const files = Array.from(dt.files).filter(f => f.type === 'application/pdf');
    await uploadFiles(files);
});

async function uploadFiles(files){
    if(!files.length) return;
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    try {
        const res = await fetch('/upload', { method: 'POST', body: form });
        const data = await res.json();
        // server returns list of saved filenames
        if (data.uploaded && data.uploaded.length) {
            uploadedFiles.push(...data.uploaded);
            renderFileList();
        }
    } catch(err){
        console.error('Upload failed', err);
    }
}

function renderFileList(){
    fileList.innerHTML = '';
    if(!uploadedFiles.length){
        const li = document.createElement('li');
        li.className = 'placeholder';
        li.innerText = 'No files uploaded yet';
        fileList.appendChild(li);
        return;
    }
    uploadedFiles.forEach((name, idx) => {
        const li = document.createElement('li');
        li.innerText = name;
        fileList.appendChild(li);
    });
}

// Q/A
askBtn.addEventListener('click', askQuestion);
questionInput.addEventListener('keydown', (e)=>{ if(e.key === 'Enter') askQuestion(); });

async function askQuestion(){
    const q = questionInput.value.trim();
    if(!q) return;
    processing.classList.remove('hidden');
    answerBox.classList.add('hidden');

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ question: q })
        });
        const payload = await res.json();
        processing.classList.add('hidden');
        if(payload.answer){
            answerText.innerHTML = payload.answer_html || payload.answer;
            answerBox.classList.remove('hidden');
            history.unshift({ question: q, answer: payload.answer });
            renderHistory();
        } else {
            answerText.innerText = 'No answer.';
            answerBox.classList.remove('hidden');
        }
    } catch(err){
        processing.classList.add('hidden');
        answerText.innerText = 'Error contacting server.';
        answerBox.classList.remove('hidden');
        console.error(err);
    }
}

function renderHistory(){
    historyList.innerHTML = '';
    if(!history.length){
        const li = document.createElement('li');
        li.className = 'placeholder';
        li.innerText = 'No search history yet';
        historyList.appendChild(li);
        return;
    }
    history.forEach(h => {
        const li = document.createElement('li');
        li.innerHTML = <strong>Q:</strong> ${escapeHtml(h.question)}<br/><small>${escapeHtml(h.answer)}</small>;
        historyList.appendChild(li);
    });
}

function escapeHtml(s){
    return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// modal events
viewHistoryBtn.addEventListener('click', ()=> historyModal.classList.remove('hidden'));
closeHistory.addEventListener('click', ()=> historyModal.classList.add('hidden'));
closeBtn.addEventListener('click', ()=> historyModal.classList.add('hidden'));
clearHistory.addEventListener('click', ()=>{
    history = [];
    renderHistory();
});