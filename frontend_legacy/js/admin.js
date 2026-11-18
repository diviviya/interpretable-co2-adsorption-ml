const token = localStorage.getItem('co2_access_token');
const authStatus = document.getElementById('auth-status');
const datasetSelect = document.getElementById('dataset-select');
const uploadButton = document.getElementById('upload-btn');
const trainButton = document.getElementById('train-btn');
const headers = () => ({'Authorization': `Bearer ${token}`});
let uploadedDataset = null;

async function api(path, options={}) {
  const response = await fetch(path, {...options, headers:{...headers(), ...(options.headers || {})}});
  const data = await response.json().catch(()=>({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
async function initialise() {
  if (!token) { authStatus.textContent = 'You are not signed in. Please sign in with an administrator account.'; return; }
  try { const me = await api('/api/auth/me'); if(me.role !== 'admin') throw new Error('This account does not have administrator privileges.'); authStatus.textContent = `Signed in as ${me.username} · Administrator access verified.`; await loadModels(); await loadAnalytics(); }
  catch(error){ authStatus.textContent = error.message; }
}
uploadButton.addEventListener('click', async () => {
  const file = document.getElementById('dataset-file').files[0]; const status = document.getElementById('upload-status');
  if(!file){status.textContent='Choose a CSV file first.';return;}
  uploadButton.disabled=true; status.textContent='Uploading and validating…';
  try { const form = new FormData(); form.append('file',file); const data=await api('/api/admin/datasets',{method:'POST',body:form}); uploadedDataset=data.report.stored_filename; datasetSelect.innerHTML=''; const option=document.createElement('option'); option.value=uploadedDataset; option.textContent=`${data.report.filename} (${data.report.row_count} rows)`; datasetSelect.append(option); trainButton.disabled=false; status.textContent=`Validated ${data.report.row_count} rows. Missing feature values are permitted and will be imputed during training.`; }
  catch(error){status.textContent=error.message;} finally{uploadButton.disabled=false;}
});
trainButton.addEventListener('click', async () => {
  const status=document.getElementById('training-status'); if(!uploadedDataset){status.textContent='Upload a dataset first.';return;}
  trainButton.disabled=true; status.textContent='Starting training…';
  try { const started=await api('/api/admin/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_name:uploadedDataset})}); const jobId=started.job.job_id; let done=false;
    while(!done){ await new Promise(resolve=>setTimeout(resolve,1800)); const job=await api(`/api/admin/training/${jobId}`); status.textContent=`${job.status.toUpperCase()} · ${job.progress_pct}% · ${job.stage || ''}`; if(job.status==='completed'){status.textContent=`Training complete. Candidate: ${job.candidate_version}\nValidation metrics: ${JSON.stringify(job.metrics,null,2)}\nThis is a validation result, not an independent test result.`;done=true;await loadModels();} else if(job.status==='failed'){throw new Error(job.error || 'Training failed');} }
  } catch(error){status.textContent=error.message;} finally{trainButton.disabled=false;}
});
async function loadModels(){
  const container=document.getElementById('models-list');
  try {const data=await api('/api/admin/models');container.innerHTML='';
    data.models.forEach(model=>{const row=document.createElement('div');row.className='model-row';const info=document.createElement('div');const name=document.createElement('b');name.textContent=model.name || model.version;const detail=document.createElement('small');detail.textContent=`${model.version} · ${model.status} · ${model.metrics ? `R² ${model.metrics.R2 ?? 'N/A'} / RMSE ${model.metrics.RMSE ?? 'N/A'}` : 'Metrics unavailable'}`;info.append(name,detail);const action=document.createElement('div');const badge=document.createElement('span');badge.className='pill';badge.textContent=model.status.toUpperCase();action.append(badge);
      if(model.status!=='active' && model.artifact_path){const button=document.createElement('button');button.textContent='Activate';button.style.marginLeft='10px';button.addEventListener('click',async()=>{if(!confirm(`Activate ${model.version}? New predictions will use this model.`))return;try{await api(`/api/admin/models/${encodeURIComponent(model.version)}/activate`,{method:'POST'});await loadModels();}catch(e){alert(e.message);}});action.append(button);} row.append(info,action);container.append(row);});
  }catch(error){container.textContent=error.message;}
}
initialise();


async function loadAnalytics(){
  const status=document.getElementById('analytics-status');
  const content=document.getElementById('analytics-content');
  if(!status || !content) return;
  status.textContent='Fetching current project statistics…';
  try{
    const data=await api('/api/admin/analytics');
    const d=data.dataset, a=data.activity, m=data.models;
    content.innerHTML='';
    const cards=document.createElement('div');
    cards.style.cssText='display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:14px 0 22px';
    const addCard=(label,value)=>{const el=document.createElement('div');el.style.cssText='background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:14px';const small=document.createElement('small');small.textContent=label;small.style.cssText='display:block;color:var(--muted);font-size:12px';const strong=document.createElement('strong');strong.textContent=value;strong.style.cssText='display:block;font-size:23px;margin-top:5px';el.append(small,strong);cards.append(el);};
    addCard('Dataset rows',d.available?d.row_count:'Unavailable');
    addCard('Features',d.feature_count);
    addCard('Predictions recorded',a.prediction_count);
    addCard('Training jobs',a.training_job_count);
    addCard('Uploaded datasets',a.uploaded_dataset_count);
    addCard('Model versions',m.registered_versions);
    content.append(cards);
    const heading=(text)=>{const h=document.createElement('h3');h.textContent=text;h.style.cssText='font-size:15px;margin:18px 0 9px';content.append(h);};
    const table=(headers,rows)=>{const t=document.createElement('table');t.style.cssText='width:100%;border-collapse:collapse;font-size:12px';const thead=document.createElement('thead');const hr=document.createElement('tr');headers.forEach(x=>{const th=document.createElement('th');th.textContent=x;th.style.cssText='text-align:left;padding:8px;border-bottom:1px solid var(--line)';hr.append(th)});thead.append(hr);t.append(thead);const tb=document.createElement('tbody');rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(x=>{const td=document.createElement('td');td.textContent=x??'—';td.style.cssText='padding:8px;border-bottom:1px solid var(--line)';tr.append(td)});tb.append(tr)});t.append(tb);content.append(t);};
    heading('CO₂ uptake distribution');
    if(d.target_stats){const st=d.target_stats;table(['Mean','Median','Minimum','Maximum','Std. deviation','Missing'],[[st.mean,st.median,st.min,st.max,st.std,st.missing].map(String)]);}
    if(d.target_distribution?.length){const max=Math.max(...d.target_distribution.map(x=>x.count),1);const chart=document.createElement('div');chart.style.cssText='display:grid;gap:8px;margin:12px 0 20px';d.target_distribution.forEach(bin=>{const row=document.createElement('div');row.style.cssText='display:grid;grid-template-columns:minmax(130px,1fr) 3fr 38px;gap:9px;align-items:center;font-size:12px';const label=document.createElement('span');label.textContent=bin.range;const track=document.createElement('div');track.style.cssText='height:12px;background:var(--soft);border-radius:8px;overflow:hidden';const bar=document.createElement('div');bar.style.cssText=`height:100%;width:${bin.count/max*100}%;background:var(--green);border-radius:8px`;track.append(bar);const count=document.createElement('span');count.textContent=bin.count;row.append(label,track,count);chart.append(row)});content.append(chart);}
    heading('Feature data quality');
    table(['Feature','Missing values','Mean','Minimum','Maximum'],(d.feature_quality||[]).map(x=>[x.feature,x.missing,x.mean,x.min,x.max].map(v=>String(v??'—'))));
    heading('System and model status');
    table(['Metric','Current value'],[['Active model',m.active_version||'Not set'],['Candidate models',m.candidate_versions],['Completed training jobs',a.completed_training_jobs],['Failed training jobs',a.failed_training_jobs]]);
    heading('Recent administrator activity');
    if(data.recent_activity?.length)table(['Time','Administrator','Action','Details'],data.recent_activity.map(x=>[x.timestamp,x.username,x.action,x.details]));
    else {const empty=document.createElement('p');empty.textContent='No administrator activity recorded yet.';content.append(empty);}
    status.textContent=d.available?'Analytics updated from the current project data and activity records.':'Activity loaded; project dataset file was not found.';
  }catch(error){status.textContent=`Could not load analytics: ${error.message}`;}
}
