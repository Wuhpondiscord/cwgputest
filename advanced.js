const $=id=>document.getElementById(id);
const U={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128};
const now=()=>performance.now(),round=x=>Math.round(x*1000)/1000;
const stats=a=>{if(!a.length)return null;const s=[...a].sort((x,y)=>x-y);return {median:round((s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2),mean:round(s.reduce((x,y)=>x+y,0)/s.length),min:round(s[0]),max:round(s.at(-1)),p95:round(s[Math.ceil(s.length*.95)-1])}};
const value=(id,min,max)=>{const n=Number($(id).value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(id+' must be '+min+'–'+max);return n};
const config=()=>({elements:value('elements',256,500000),dispatches:value('dispatches',2,64),trials:value('trials',1,20),warmups:value('warmups',0,5)});
const shader=`@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> b:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<arrayLength(&b)){b[id.x]=a[id.x]+1.0;}}`;
let latest=null;
const pause=()=>new Promise(r=>setTimeout(r,0));
async function device(){if(!navigator.gpu)throw Error('WebGPU not supported');const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No adapter');const d=await adapter.requestDevice();const module=d.createShaderModule({code:shader});const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length){d.destroy();throw Error(errors.map(x=>x.message).join('; '))}const p=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});return {d,p,adapterInfo:{vendor:adapter.info?.vendor??'',architecture:adapter.info?.architecture??''}}}
function buf(d,size,usage,owned){const b=d.createBuffer({size,usage});owned.push(b);return b}
function step(d,p,source,target,n,enc){const pass=enc.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,d.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:target}}]}));pass.dispatchWorkgroups(Math.ceil(n/64));pass.end()}
async function read(d,source,offset,bytes){const staging=d.createBuffer({size:bytes,usage:U.COPY_DST|U.MAP_READ});try{const enc=d.createCommandEncoder();enc.copyBufferToBuffer(source,offset,staging,0,bytes);d.queue.submit([enc.finish()]);await staging.mapAsync(U.MAP_READ);const result=new Uint8Array(staging.getMappedRange().slice(0));staging.unmap();return result}finally{staging.destroy()}}
function init(n){const x=new Float32Array(n);for(let i=0;i<n;i++)x[i]=i%1024;return x}
function check(actual,expected,steps){const view=new Float32Array(actual.buffer,actual.byteOffset,actual.byteLength/4);let errors=0;for(let i=0;i<view.length;i++)if(view[i]!==expected[i]+steps)errors++;return errors}
async function compute(d,p,n,steps,mode,interval=0){const owned=[],bytes=n*4;try{const a=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),b=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),initial=init(n);d.queue.writeBuffer(a,0,initial);let source=a,target=b,intermediateBytes=0,checkpointMs=0;const start=now();
if(mode==='batched'){const enc=d.createCommandEncoder();for(let i=0;i<steps;i++){step(d,p,source,target,n,enc);[source,target]=[target,source]}d.queue.submit([enc.finish()])}
else for(let i=0;i<steps;i++){const enc=d.createCommandEncoder();step(d,p,source,target,n,enc);d.queue.submit([enc.finish()]);[source,target]=[target,source];if(mode==='readback-each'||(mode==='frequency'&&interval>0&&(i+1)%interval===0)){const t=now();const snapshot=await read(d,source,0,bytes);checkpointMs+=now()-t;intermediateBytes+=snapshot.byteLength}}
const final=await read(d,source,0,bytes);const elapsedMs=now()-start;const mismatches=check(final,initial,steps);return {mode,interval,elapsedMs:round(elapsedMs),checkpointMs:round(checkpointMs),intermediateBytes,finalBytes:bytes,mismatches,valid:mismatches===0}}finally{owned.forEach(x=>x.destroy())}}
async function snapshot(d,source,n,parts){const bytes=n*4,chunks=[],start=now();for(let k=0;k<parts;k++){const startEl=Math.floor(k*n/parts),endEl=Math.floor((k+1)*n/parts);if(endEl===startEl)continue;chunks.push(await read(d,source,startEl*4,(endEl-startEl)*4))}const snapshotMs=now()-start;return {chunks,snapshotMs:round(snapshotMs)}}
async function checkpoint(d,n,parts){const owned=[],bytes=n*4;try{const a=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),b=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),initial=init(n);d.queue.writeBuffer(a,0,initial);const s=await snapshot(d,a,n,parts);const t=now();let offset=0;for(const chunk of s.chunks){d.queue.writeBuffer(b,offset,chunk);offset+=chunk.byteLength}await d.queue.onSubmittedWorkDone();const restoreMs=now()-t;const v=now(),data=await read(d,b,0,bytes),verificationMs=now()-v,mismatches=check(data,initial,0);return {elements:n,bytes,parts,actualChunks:s.chunks.length,snapshotMs:s.snapshotMs,restoreMs:round(restoreMs),verificationMs:round(verificationMs),mismatches,valid:mismatches===0,snapshotMiBPerSec:round(bytes/1048576/(s.snapshotMs/1000))}}finally{owned.forEach(x=>x.destroy())}}

function ranges(n,parts){const out=[];for(let k=0;k<parts;k++){const start=Math.floor(k*n/parts)*4,end=Math.floor((k+1)*n/parts)*4;if(end>start)out.push({offset:start,bytes:end-start})}return out}
async function captureStrategies(d,source,n,parts,method){
 const rangesList=ranges(n,parts),start=now();
 let chunks=[],gpuSubmissions=0,gpuMapCalls=0,cpuSplitMs=0;
 if(method==='sequential'){
  for(const r of rangesList){chunks.push(await read(d,source,r.offset,r.bytes));gpuSubmissions++;gpuMapCalls++}
 }else if(method==='batched-copies'){
  const owned=[],stages=[];
  try{
   const enc=d.createCommandEncoder();
   for(const r of rangesList){const stage=buf(d,r.bytes,U.COPY_DST|U.MAP_READ,owned);stages.push(stage);enc.copyBufferToBuffer(source,r.offset,stage,0,r.bytes)}
   d.queue.submit([enc.finish()]);gpuSubmissions=1;
   await Promise.all(stages.map(stage=>stage.mapAsync(U.MAP_READ)));gpuMapCalls=stages.length;
   chunks=stages.map(stage=>{const copy=new Uint8Array(stage.getMappedRange().slice(0));stage.unmap();return copy});
  }finally{owned.forEach(stage=>{try{stage.unmap()}catch{}stage.destroy()})}
 }else if(method==='single-readback-cpu-split'){
  const whole=await read(d,source,0,n*4);gpuSubmissions=1;gpuMapCalls=1;
  const splitStart=now();chunks=rangesList.map(r=>whole.slice(r.offset,r.offset+r.bytes));cpuSplitMs=now()-splitStart;
 }else throw Error('Unsupported capture method');
 return {chunks,snapshotMs:round(now()-start),cpuSplitMs:round(cpuSplitMs),gpuSubmissions,gpuMapCalls};
}
async function captureComparison(d,n,parts,method){
 const owned=[],bytes=n*4;
 try{
  const original=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),restored=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),expected=init(n);
  d.queue.writeBuffer(original,0,expected);
  const captured=await captureStrategies(d,original,n,parts,method);
  const restoreStart=now();let offset=0;
  for(const chunk of captured.chunks){d.queue.writeBuffer(restored,offset,chunk);offset+=chunk.byteLength}
  await d.queue.onSubmittedWorkDone();
  const restoreMs=round(now()-restoreStart);
  const verifyStart=now(),out=await read(d,restored,0,bytes),verificationMs=round(now()-verifyStart);
  const mismatches=check(out,expected,0);
  return {method,parts,elements:n,bytes,actualChunks:captured.chunks.length,
   snapshotMs:captured.snapshotMs,cpuSplitMs:captured.cpuSplitMs,gpuSubmissions:captured.gpuSubmissions,
   gpuMapCalls:captured.gpuMapCalls,restoreMs,verificationMs,mismatches,valid:mismatches===0,
   snapshotMiBPerSec:round(bytes/1048576/(captured.snapshotMs/1000))};
 }finally{owned.forEach(b=>b.destroy())}
}

async function bench(label,runner){const c=config();$('status').textContent='Running '+label+'...';const {d,p,adapterInfo}=await device();try{if(c.elements*4>d.limits.maxStorageBufferBindingSize||c.elements*4>d.limits.maxBufferSize)throw Error('Requested buffer exceeds GPU limits');const records=[];for(let i=0;i<c.warmups+c.trials;i++){const rows=await runner(d,p,c,i);if(i>=c.warmups)for(const r of rows)records.push({...r,trial:i-c.warmups+1});$('status').textContent=label+': '+Math.max(0,i-c.warmups+1)+'/'+c.trials+' measured rounds';await pause()}const groups={};for(const r of records){const key=r.label??r.mode??String(r.parts);(groups[key]??=[]).push(r)}const summary={};for(const [key,rows] of Object.entries(groups))summary[key]={elapsedMs:stats(rows.map(r=>r.elapsedMs).filter(x=>x!==undefined)),snapshotMs:stats(rows.map(r=>r.snapshotMs).filter(x=>x!==undefined)),restoreMs:stats(rows.map(r=>r.restoreMs).filter(x=>x!==undefined)),checkpointMs:stats(rows.map(r=>r.checkpointMs).filter(x=>x!==undefined)),valid:rows.every(r=>r.valid),sampleCount:rows.length};latest={version:2,kind:label,createdAt:new Date().toISOString(),configuration:c,adapterInfo,summary,records,notes:['Wall-clock timings include browser scheduling and synchronization.','Restore time includes queued GPU writes and onSubmittedWorkDone.','Same-device checkpoints are not live cross-device migration.','Throughput figures include per-chunk overhead.']};$('summary').textContent=JSON.stringify(summary,null,2);$('details').textContent=JSON.stringify(records,null,2);$('export').disabled=false;$('status').textContent='Completed '+label+'. All valid: '+records.every(r=>r.valid)}finally{d.destroy()}}
async function execute(button,fn){button.disabled=true;try{await fn()}catch(e){$('status').textContent='Failed: '+String(e);console.error(e)}finally{button.disabled=false}}
$('batch').onclick=()=>execute($('batch'),()=>bench('dispatch comparison',async(d,p,c,i)=>{const modes=i%2?['readback-each','batched','separate']:['separate','batched','readback-each'];const rows=[];for(const mode of modes)rows.push({...await compute(d,p,c.elements,c.dispatches,mode),label:mode});return rows}));
$('chunks').onclick=()=>execute($('chunks'),()=>bench('chunk scaling',async(d,p,c,i)=>{const parts=i%2?[64,16,4,1]:[1,4,16,64];const rows=[];for(const count of parts)rows.push({...await checkpoint(d,c.elements,count),label:count+' chunks'});return rows}));
$('sizes').onclick=()=>execute($('sizes'),()=>bench('state scaling',async(d,p,c,i)=>{const sizes=[65536,131072,262144,500000];if(i%2)sizes.reverse();const rows=[];for(const n of sizes){if(n*4>d.limits.maxStorageBufferBindingSize)throw Error('GPU buffer limit too small for size test');rows.push({...await checkpoint(d,n,4),label:n+' elements'})}return rows}));
$('frequency').onclick=()=>execute($('frequency'),()=>bench('checkpoint frequency',async(d,p,c,i)=>{const intervals=[0,16,4,1];if(i%2)intervals.reverse();const rows=[];for(const interval of intervals)rows.push({...await compute(d,p,c.elements,c.dispatches,'frequency',interval),label:interval===0?'none':'every '+interval});return rows}));
$('capturecompare').onclick=()=>execute($('capturecompare'),()=>bench('checkpoint capture strategy comparison',async(d,p,c,i)=>{
 const methods=['sequential','batched-copies','single-readback-cpu-split'];
 const cases=[];for(const parts of [1,4,16,64])for(const method of methods)cases.push({parts,method});
 const rotated=cases.slice(i%cases.length).concat(cases.slice(0,i%cases.length));
 if(i%2)rotated.reverse();
 const rows=[];
 for(const item of rotated){rows.push({...await captureComparison(d,c.elements,item.parts,item.method),label:item.method+' / '+item.parts+' chunks'})}
 return rows;
}));


async function frequencyMeasured(d,p,n,steps,interval){
 const owned=[],bytes=n*4,expected=init(n);let source,target,checkpointCount=0,checkpointMs=0,checkpointBytes=0;
 try{
  source=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned);
  target=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned);
  d.queue.writeBuffer(source,0,expected);
  const start=now();
  for(let i=0;i<steps;i++){
   const enc=d.createCommandEncoder();step(d,p,source,target,n,enc);d.queue.submit([enc.finish()]);
   [source,target]=[target,source];
   // Do not count a snapshot at the final dispatch: a final readback is common to every mode.
   if(interval>0&&(i+1)%interval===0&&i+1<steps){
    const t=now(),data=await read(d,source,0,bytes);
    checkpointMs+=now()-t;checkpointCount++;checkpointBytes+=data.byteLength;
    const cpMismatches=check(data,expected,i+1);
    if(cpMismatches)throw Error('Intermediate checkpoint validation failed after dispatch '+(i+1));
   }
  }
  const final=await read(d,source,0,bytes),elapsedMs=now()-start,mismatches=check(final,expected,steps);
  return {interval,dispatches:steps,checkpointCount,checkpointBytes,
   checkpointMs:round(checkpointMs),elapsedMs:round(elapsedMs),overheadScope:'Intermediate snapshots only; final readback is common',
   mismatches,valid:mismatches===0};
 }finally{owned.forEach(b=>b.destroy())}
}
async function restoreAndContinue(){
 const file=$('import').files?.[0];
 if(!file)throw Error('Select a portable checkpoint JSON file first');
 if(file.size>8000000)throw Error('Checkpoint exceeds the 8 MB limit');
 const cp=JSON.parse(await file.text());
 if(cp.format!=='cwgpu-portable-checkpoint'||cp.version!==1||cp.shader!=='vector-add-one-v1'||cp.encoding!=='float32-le-base64')
  throw Error('Unknown checkpoint format or shader');
 if(!Number.isSafeInteger(cp.elements)||cp.elements<1||cp.elements>500000||
    !Number.isSafeInteger(cp.completedDispatches)||cp.completedDispatches<0||cp.completedDispatches>100000)
  throw Error('Invalid checkpoint configuration');
 const extra=value('resumeDispatches',1,64);
 const raw=unbase64(cp.data);
 if(raw.byteLength!==cp.elements*4||await digest(raw)!==cp.sha256)throw Error('Checkpoint size or hash mismatch');
 const {d,p,adapterInfo}=await device(),owned=[];
 try{
  if(raw.byteLength>d.limits.maxStorageBufferBindingSize||raw.byteLength>d.limits.maxBufferSize)
   throw Error('Destination device storage limit exceeded');
  const sourceBuffer=buf(d,raw.byteLength,U.STORAGE|U.COPY_SRC|U.COPY_DST,owned),
        targetBuffer=buf(d,raw.byteLength,U.STORAGE|U.COPY_SRC|U.COPY_DST,owned);
  const restoreStart=now();d.queue.writeBuffer(sourceBuffer,0,raw);await d.queue.onSubmittedWorkDone();
  const restoreMs=now()-restoreStart;
  const initial=init(cp.elements);
  // Validate the actual restored GPU state before resuming execution.
  const before=await read(d,sourceBuffer,0,raw.byteLength);
  const restoreMismatches=check(before,initial,cp.completedDispatches);
  if(restoreMismatches)throw Error('Restored contents failed GPU readback validation');
  let source=sourceBuffer,target=targetBuffer;
  const start=now(),enc=d.createCommandEncoder();
  for(let i=0;i<extra;i++){step(d,p,source,target,cp.elements,enc);[source,target]=[target,source]}
  d.queue.submit([enc.finish()]);
  const final=await read(d,source,0,raw.byteLength),executionMs=now()-start,
        mismatches=check(final,initial,cp.completedDispatches+extra);
  const result={kind:'portable-checkpoint-resume',createdAt:new Date().toISOString(),
   sourceAdapter:cp.sourceAdapter,destinationAdapter:adapterInfo,elements:cp.elements,
   checkpointHash:cp.sha256,priorDispatches:cp.completedDispatches,resumedDispatches:extra,
   totalDispatches:cp.completedDispatches+extra,restoreMs:round(restoreMs),
   executionAndFinalReadbackMs:round(executionMs),restoreMismatches,mismatches,valid:mismatches===0};
  $('status').textContent=result.valid?'Checkpoint restored, GPU execution resumed, output validated.':'Resumed computation failed validation.';
  $('summary').textContent=JSON.stringify(result,null,2);
  $('details').textContent='This resumes a deterministic vector-add workload from a saved GPU buffer; it is not arbitrary live kernel-state migration.';
  latest={version:3,kind:'portable-checkpoint-resume',createdAt:result.createdAt,
    configuration:{elements:cp.elements,priorDispatches:cp.completedDispatches,resumedDispatches:extra},
    adapterInfo,summary:result,records:[result]};
  $('export').disabled=false;
 }finally{owned.forEach(b=>b.destroy());d.destroy()}
}
$('sizecompare').onclick=()=>execute($('sizecompare'),()=>bench('state size by capture strategy',async(d,p,c,i)=>{
 const sizes=[65536,131072,262144,500000],rows=[];
 if(i%2)sizes.reverse();
 for(const n of sizes){
  if(n*4>d.limits.maxStorageBufferBindingSize||n*4>d.limits.maxBufferSize)throw Error('Device cannot allocate '+n+' elements');
  const methods=i%2?['single-readback-cpu-split','batched-copies','sequential']:['sequential','batched-copies','single-readback-cpu-split'];
  for(const method of methods){
   const parts=method==='sequential'?1:16;
   rows.push({...await captureComparison(d,n,parts,method),
    label:n+' elements / '+method+(method==='sequential'?' (1 chunk)':' (16 chunks)')});
  }
 }
 return rows;
}));
$('frequency2').onclick=()=>execute($('frequency2'),()=>bench('checkpoint frequency with intermediate validation',async(d,p,c,i)=>{
 const intervals=[0,16,4,1],rows=[];if(i%2)intervals.reverse();
 for(const interval of intervals)rows.push({...await frequencyMeasured(d,p,c.elements,c.dispatches,interval),
    label:interval===0?'none':('every '+interval)});
 return rows;
}));
$('resume').onclick=()=>execute($('resume'),restoreAndContinue);

function download(name,body,type){const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)}
function base64(bytes){let result='';for(let i=0;i<bytes.length;i+=8192)result+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(result)}
function unbase64(s){const raw=atob(s),out=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)out[i]=raw.charCodeAt(i);return out}
async function digest(bytes){const hash=new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));return Array.from(hash,x=>x.toString(16).padStart(2,'0')).join('')}
$('make').onclick=()=>execute($('make'),async()=>{const c=config(),{d,p,adapterInfo}=await device();try{const owned=[],bytes=c.elements*4;if(bytes>d.limits.maxStorageBufferBindingSize)throw Error('Buffer limit exceeded');const a=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned),b=buf(d,bytes,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned);try{const initial=init(c.elements);d.queue.writeBuffer(a,0,initial);let src=a,dst=b;for(let i=0;i<c.dispatches;i++){const enc=d.createCommandEncoder();step(d,p,src,dst,c.elements,enc);d.queue.submit([enc.finish()]);[src,dst]=[dst,src]}const raw=await read(d,src,0,bytes),mismatches=check(raw,initial,c.dispatches);if(mismatches)throw Error('Checkpoint source invalid: '+mismatches+' mismatches');const obj={format:'cwgpu-portable-checkpoint',version:1,createdAt:new Date().toISOString(),elements:c.elements,completedDispatches:c.dispatches,shader:'vector-add-one-v1',encoding:'float32-le-base64',sha256:await digest(raw),data:base64(raw),sourceAdapter:adapterInfo};download('cwgpu-checkpoint-'+Date.now()+'.json',JSON.stringify(obj),'application/json');$('status').textContent='Exported '+bytes+' bytes of verified GPU state; SHA-256 '+obj.sha256}catch(e){throw e}finally{owned.forEach(x=>x.destroy())}}finally{d.destroy()}});
$('restore').onclick=()=>execute($('restore'),async()=>{const file=$('import').files?.[0];if(!file)throw Error('Choose a checkpoint JSON file');if(file.size>8000000)throw Error('Checkpoint JSON exceeds 8 MB safety limit');const cp=JSON.parse(await file.text());if(cp.format!=='cwgpu-portable-checkpoint'||cp.version!==1||cp.shader!=='vector-add-one-v1'||cp.encoding!=='float32-le-base64')throw Error('Unsupported checkpoint format');if(!Number.isSafeInteger(cp.elements)||cp.elements<1||cp.elements>500000||!Number.isSafeInteger(cp.completedDispatches)||cp.completedDispatches<0||cp.completedDispatches>100000)throw Error('Invalid checkpoint dimensions');const raw=unbase64(cp.data);if(raw.byteLength!==cp.elements*4)throw Error('Checkpoint byte count mismatch');const hash=await digest(raw);if(hash!==cp.sha256)throw Error('Checkpoint SHA-256 mismatch');const {d,adapterInfo}=await device();try{if(raw.byteLength>d.limits.maxStorageBufferBindingSize)throw Error('Destination GPU limit exceeded');const owned=[],gpu=buf(d,raw.byteLength,U.STORAGE|U.COPY_DST|U.COPY_SRC,owned);try{const start=now();d.queue.writeBuffer(gpu,0,raw);await d.queue.onSubmittedWorkDone();const restoreMs=now()-start,verifyStart=now(),out=await read(d,gpu,0,raw.byteLength),verificationMs=now()-verifyStart,initial=init(cp.elements),mismatches=check(out,initial,cp.completedDispatches);const result={sourceAdapter:cp.sourceAdapter,destinationAdapter:adapterInfo,elements:cp.elements,completedDispatches:cp.completedDispatches,sha256:hash,restoreMs:round(restoreMs),verificationMs:round(verificationMs),mismatches,valid:mismatches===0};$('status').textContent=result.valid?'Portable checkpoint restored and validated.':'Checkpoint restore failed validation.';$('summary').textContent=JSON.stringify(result,null,2);$('details').textContent='Import verifies SHA-256 and exact GPU float32 output. This is manual cross-browser buffer restoration, not live execution migration.'}finally{owned.forEach(x=>x.destroy())}}finally{d.destroy()}});
$('export').onclick=()=>{if(latest)download('cwgpu-advanced-'+Date.now()+'.json',JSON.stringify(latest,null,2),'application/json')};
