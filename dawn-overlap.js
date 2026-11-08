// Bounded, browser-local checkpoint pipeline experiment (not native GPU-state migration).
const O=id=>document.getElementById(id);
const USAGE={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128};
const stamp=()=>performance.now(),r3=x=>Math.round(x*1000)/1000;
function summarize(xs){if(!xs.length)return null;const a=xs.slice().sort((x,y)=>x-y);return {median:r3(a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2),mean:r3(a.reduce((x,y)=>x+y,0)/a.length),min:r3(a[0]),max:r3(a.at(-1))}}
const W=`@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> outputState:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x<arrayLength(&outputState)){outputState[id.x]=source[id.x]+1u;}
}`;
function initial(n){const a=new Uint32Array(n);for(let i=0;i<n;i++)a[i]=i%8192;return a}
function verify(stage,n,steps){const bytes=stage.getMappedRange(),a=new Uint32Array(bytes);for(let i=0;i<n;i++)if(a[i]!==((i%8192)+steps))throw Error('checkpoint mismatch at '+i+' after '+steps+' steps')}
async function experiment(d,p,c,mode){
 const n=c.elements,bytes=n*4,stages=[],owned=[],records=[],pending=[],state=initial(n);
 const make=(size,usage)=>{const b=d.createBuffer({size,usage});owned.push(b);return b};
 let outstandingPeak=0,waitMs=0,submitCount=0,source,target;
 async function drain(slot){
  const t=stamp();
  await slot.promise;
  waitMs+=stamp()-t;
  verify(slot.buffer,n,slot.steps);
  slot.buffer.unmap();
  slot.finishedAt=stamp();
  records.push({afterDispatch:slot.steps,copySubmitToMappedMs:r3(slot.finishedAt-slot.startedAt),valid:true});
  slot.occupied=false;
 }
 try{
  source=make(bytes,USAGE.STORAGE|USAGE.COPY_DST|USAGE.COPY_SRC);
  target=make(bytes,USAGE.STORAGE|USAGE.COPY_SRC|USAGE.COPY_DST);
  d.queue.writeBuffer(source,0,state);
  for(let i=0;i<c.slots;i++)stages.push({buffer:make(bytes,USAGE.MAP_READ|USAGE.COPY_DST),occupied:false});
  const started=stamp();
  for(let i=1;i<=c.dispatches;i++){
   const enc=d.createCommandEncoder();
   const pass=enc.beginComputePass();pass.setPipeline(p);
   pass.setBindGroup(0,d.createBindGroup({layout:p.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:target}}]}));
   pass.dispatchWorkgroups(Math.ceil(n/64));pass.end();
   const wantsSnapshot=i<c.dispatches&&i%c.interval===0;
   let slot=null;
   if(wantsSnapshot){
    slot=stages.find(s=>!s.occupied);
    if(!slot){
     // Bounded backpressure; waiting for the oldest snapshot frees one staging buffer.
     const oldest=pending.shift();
     if(!oldest)throw Error('staging pool accounting error');
     await drain(oldest);
     slot=oldest;
    }
    slot.occupied=true;slot.steps=i;slot.startedAt=stamp();
    enc.copyBufferToBuffer(target,0,slot.buffer,0,bytes);
   }
   d.queue.submit([enc.finish()]);submitCount++;
   [source,target]=[target,source];
   if(slot){
    slot.promise=slot.buffer.mapAsync(USAGE.MAP_READ);
    if(mode==='blocking')await drain(slot);
    else{
     pending.push(slot);
     outstandingPeak=Math.max(outstandingPeak,pending.length);
    }
   }
  }
  for(const item of pending)await drain(item);
  const stage=make(bytes,USAGE.MAP_READ|USAGE.COPY_DST),e=d.createCommandEncoder();
  e.copyBufferToBuffer(source,0,stage,0,bytes);d.queue.submit([e.finish()]);submitCount++;
  await stage.mapAsync(USAGE.MAP_READ);verify(stage,n,c.dispatches);stage.unmap();
  const elapsedMs=r3(stamp()-started);
  return {mode,elapsedMs,checkpointCount:records.length,checkpointBytes:records.length*bytes,
   gpuSubmissions:submitCount,peakPendingMaps:outstandingPeak,waitMs:r3(waitMs),
   snapshotCompletionMs:summarize(records.map(x=>x.copySubmitToMappedMs)),
   checkpoints:records.sort((a,b)=>a.afterDispatch-b.afterDispatch),valid:records.every(x=>x.valid)};
 }finally{
  // Pending GPU maps need to be settled/cancelled before a device is reused.
  for(const slot of pending){if(slot.occupied){try{slot.buffer.unmap()}catch{};if(slot.promise)await slot.promise.catch(()=>{})}}
  for(const b of owned){try{b.unmap()}catch{}b.destroy()}
 }
}
let overlapReport=null;
const yieldBrowser=()=>new Promise(r=>setTimeout(r,0));
O('overlapRun').onclick=async()=>{
 O('overlapRun').disabled=true;O('overlapExport').disabled=true;
 const c={elements:Number(O('overlapElements').value),dispatches:Number(O('overlapSteps').value),
 interval:Number(O('overlapInterval').value),slots:Number(O('overlapSlots').value),
 trials:Number(O('overlapTrials').value),warmups:Number(O('overlapWarmups').value)};
 overlapReport={kind:'dawn-checkpoint-overlap',version:1,startedAt:new Date().toISOString(),config:c,
 records:[],errors:[],notes:[
 'Blocking mode waits for every intermediate map; overlapped mode enqueues subsequent work while maps remain pending.',
 'Both modes submit one compute command buffer per dispatch and the same checkpoint bytes.',
 'Overlapped mode uses a fixed staging-buffer pool with backpressure, not unbounded allocations.',
 'Includes dispatch, copy, map completion, CPU verification, allocations and final readback; not pure GPU time.',
 'No transfer to another device and no proof of true simultaneous GPU copy/compute execution.'
 ]};
 let d;
 try{
  for(const [k,min,max] of [['elements',256,500000],['dispatches',8,64],['interval',1,32],['slots',1,4],['trials',1,20],['warmups',0,5]])
   if(!Number.isSafeInteger(c[k])||c[k]<min||c[k]>max)throw Error(k+' outside '+min+'–'+max);
  if(c.interval>=c.dispatches)throw Error('Interval must be below dispatch count');
  if(!navigator.gpu)throw Error('WebGPU unavailable');
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No GPU adapter');
  d=await adapter.requestDevice();if(c.elements*4>d.limits.maxStorageBufferBindingSize||c.elements*4>d.limits.maxBufferSize)throw Error('Storage buffer limit');
  const mod=d.createShaderModule({code:W});
  const errors=(await mod.getCompilationInfo()).messages.filter(x=>x.type==='error');
  if(errors.length)throw Error(errors.map(x=>x.message).join('; '));
  const pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module:mod,entryPoint:'main'}});
  overlapReport.adapter={vendor:adapter.info?.vendor??'',architecture:adapter.info?.architecture??'',userAgent:navigator.userAgent};
  for(let i=0;i<c.trials+c.warmups;i++){
   const modes=i%2===0?['blocking','overlapped']:['overlapped','blocking'];
   for(const mode of modes){
    O('overlapStatus').textContent='Trial '+(i+1)+'/'+(c.trials+c.warmups)+': '+mode;
    try{
     const result=await experiment(d,pipeline,c,mode);
     if(i>=c.warmups)overlapReport.records.push({...result,trial:i-c.warmups+1});
     if(!result.valid)throw Error('Checkpoint mismatch');
    }catch(e){overlapReport.errors.push({trial:i+1,mode,error:String(e)});throw e}
    await yieldBrowser();
   }
  }
  overlapReport.summary={};
  for(const mode of ['blocking','overlapped']){
   const rows=overlapReport.records.filter(x=>x.mode===mode);
   overlapReport.summary[mode]={elapsedMs:summarize(rows.map(x=>x.elapsedMs)),
    waitMs:summarize(rows.map(x=>x.waitMs)),peakPendingMaps:Math.max(0,...rows.map(x=>x.peakPendingMaps)),
    allValid:rows.every(x=>x.valid)};
  }
  overlapReport.summary.medianSpeedup=r3(overlapReport.summary.blocking.elapsedMs.median/
   overlapReport.summary.overlapped.elapsedMs.median);
  O('overlapStatus').textContent='Complete: '+overlapReport.records.length+' measured runs; all checkpoint contents verified.';
 }catch(e){overlapReport.errors.push({stage:'setup-or-execution',error:String(e)});O('overlapStatus').textContent='Stopped: '+String(e)+' — partial JSON export available'}
 finally{d?.destroy();overlapReport.finishedAt=new Date().toISOString();
  O('overlapResults').textContent=JSON.stringify(overlapReport,null,2);O('overlapExport').disabled=false;O('overlapRun').disabled=false}
};
O('overlapExport').onclick=()=>{if(!overlapReport)return;const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(overlapReport,null,2)],{type:'application/json'}));a.href=url;a.download='dawn-overlap-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)};
