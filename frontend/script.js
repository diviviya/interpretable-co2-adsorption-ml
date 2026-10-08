const form=document.getElementById("form"),result=document.getElementById("result"),predict=document.getElementById("predict");
const example={surface_area:1420,total_pore_volume:.74,micropore_volume:.48,carbon_pct:78.855,hydrogen_pct:1.5,nitrogen_pct:2.26,oxygen_pct:15.05,sulfur_pct:0,temperature_c:25,pressure_bar:1};
document.getElementById("example").onclick=()=>Object.entries(example).forEach(([k,v])=>form.elements[k].value=v);
document.getElementById("clear").onclick=()=>{form.reset();placeholder()};
form.onsubmit=async e=>{
 e.preventDefault(); const data=Object.fromEntries(new FormData(form)); Object.keys(data).forEach(k=>data[k]=Number(data[k]));
 if(Object.values(data).some(v=>!Number.isFinite(v))){alert("Please enter valid numerical values.");return}
 predict.disabled=true; predict.firstChild.textContent="Predicting..."; await new Promise(r=>setTimeout(r,600));

 
 const demo=.001*data.surface_area+.70*data.pressure_bar-.018*data.temperature_c+.25*data.micropore_volume+.10*data.total_pore_volume;
 const value=Math.max(.1,Math.min(8.2,demo));
 result.innerHTML=`<div><div class="prediction">${value.toFixed(3)}</div><div class="unit">mmol/g</div><div class="label">Predicted CO₂ uptake</div></div>`;
 predict.disabled=false; predict.firstChild.textContent="Predict CO₂ uptake";
};
function placeholder(){result.innerHTML='<div><div class="icon">↗</div><h3>Your prediction will appear here</h3><p>Fill the features and click the prediction button.</p></div>'}