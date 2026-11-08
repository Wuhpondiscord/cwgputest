const $=id=>document.getElementById(id);
const U={MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,STORAGE:128};
let last=null;
const now=()=>performance.now(),round=n=>Math.round(n*1000)/1000;
const stat=arr=>{if(!arr.length)return null;const a=[...arr].sort((x,y)=>x-y);return {median:round((a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2),mean:round(a.reduce((x,y)=>x+y,0)/a.length),min:round(a[0]),max:round(a.at(-1)),p95:round(a[Math.ceil(a.length*.95)-1])}};
const input=(id,min,max)=>{const v=Number($(id).value);if(!Number.isSafeInteger(v)||v<min||v>max)throw Error(id+' must be '+min+'–'+max);return v};
const shader=`
@group(0) @binding(0) var<storage,read> inputData:array<f32>;
@group(0) @binding(1) var<storage,read_write> outputData:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x<arrayLength(&outputData)){outputData[id.x]=inputData[id.x]+1.0;}
}`;
const yieldUI=()=>new Promise(resolve=>setTimeout(resolve,0));
async function create(d){const module=d.createShaderModule({code:shader});const info=await module.getCompilationInfo();const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(x=>x.message).join('; '));return d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}})}
function allocate(d,n,resources){const b=d.createBuffer({size:n,usage:U.STORAGE|U.COPY_SRC|U.COPY_DST});resources.push(b);return b}
function dispatch(d,p,source,target,n,readback){const enc=d.createCommandEncoder();const group=d.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:target}}]});const pass=enc.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(n/64));pass.end();if(readback)enc.copyBufferToBuffer(target,0,readback,0,n*4);d.queue.submit([enc.finish()])}
async function read(d,source,bytes){const staging=d.createBuffer({size:bytes,usage:U.COPY_DST|U.MAP_READ});try{const enc=d.createCommandEncoder();enc.copyBufferToBuffer(source,0,staging,0,bytes);d.queue.submit([enc.finish()]);await staging.mapAsync(1);const out=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();return out}finally{staging.destroy()}}
async function residency(d,p,n,steps,mode){const resources=[],bytes=n*4;try{
 const a=allocate(d,bytes,resources),b=allocate(d,bytes,resources),initial=new Float32Array(n);for(let i=0;i<n;i++)initial[i]=i%1024;
 d.queue.writeBuffer(a,0,initial);const t=now();let source=a,target=b;let intermediateBytes=0;
 for(let i=0;i<steps;i++){
 dispatch(d,p,source,target,n);
 if(mode==='readback-each'){const snapshot=await read(d,target,bytes);intermediateBytes+=snapshot.byteLength;}
 [source,target]=[target,source];
 }
 const final=await read(d,source,bytes),elapsed=now()-t;let mismatches=0;
 for(let i=0;i<n;i++)if(final[i]!==initial[i]+steps)mismatches++;
 return {mode,elapsedMs:round(elapsed),mismatches,valid:mismatches===0,dispatches:steps,intermediateBytesRead:intermediateBytes,finalBytesRead:final.byteLength,throughputElementsPerSecond:round(n*steps/(elapsed/1000))};
 }finally{resources.forEach(x=>x.destroy())}}
async function checkpoint(d,n,chunkElements){const resources=[],bytes=n*4;try{
 const original=allocate(d,bytes,resources),restored=allocate(d,bytes,resources),values=new Float32Array(n);for(let i=0;i<n;i++)values[i]=(i%997)/997;
 d.queue.writeBuffer(original,0,values);
 const start=now(),chunks=[];let readbackMs=0,serializeMs=0;
 for(let offset=0;offset<n;offset+=chunkElements){
 const length=Math.min(chunkElements,n-offset),chunkBytes=length*4,stage=d.createBuffer({size:chunkBytes,usage:U.COPY_DST|U.MAP_READ});
 try{const t=now(),enc=d.createCommandEncoder();enc.copyBufferToBuffer(original,offset*4,stage,0,chunkBytes);d.queue.submit([enc.finish()]);await stage.mapAsync(1);const copy=new Uint8Array(stage.getMappedRange().slice(0));stage.unmap();readbackMs+=now()-t;
 const s=now();chunks.push(copy);serializeMs+=now()-s;
 }finally{stage.destroy()}
 }
 const snapshotMs=now()-start;
 const restoreStart=now();let offset=0;for(const chunk of chunks){d.queue.writeBuffer(restored,offset,chunk);offset+=chunk.byteLength}await d.queue.onSubmittedWorkDone();const restoreMs=now()-restoreStart;
 const checkStart=now(),out=await read(d,restored,bytes);let mismatches=0;for(let i=0;i<n;i++)if(out[i]!==values[i])mismatches++;const verificationMs=now()-checkStart;
 return {valid:mismatches===0,mismatches,bytes,chunks:chunks.length,chunkElements,readbackMs:round(readbackMs),serializeMs:round(serializeMs),snapshotMs:round(snapshotMs),restoreMs:round(restoreMs),verificationMs:round(verificationMs),snapshotMiBPerSec:round(bytes/(1024*1024)/(snapshotMs/1000)),restoreMiBPerSec:round(bytes/(1024*1024)/(restoreMs/1000)),note:'Restore is to a second GPUBuffer on the same device; not another browser or GPU.'};
 }finally{resources.forEach(x=>x.destroy())}}
$('run').onclick=async()=>{const btn=$('run');btn.disabled=true;$('export').disabled=true;try{
 const n=input('elements',1,500000),steps=input('dispatches',2,64),trials=input('trials',1,20),warmups=input('warmups',0,5),chunk=input('chunk',256,500000);
 if(!navigator.gpu)throw Error('WebGPU is unavailable');
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No WebGPU adapter');
 const d=await adapter.requestDevice();const bytes=n*4;if(bytes>d.limits.maxStorageBufferBindingSize||bytes>d.limits.maxBufferSize)throw Error('Exceeds adapter buffer limits');
 try{
 const pipeline=await create(d),residencyRuns=[],checkpointRuns=[];
 for(let i=0;i<warmups+trials;i++){
 const order=i%2?['readback-each','gpu-resident']:['gpu-resident','readback-each'];
 for(const mode of order){const result=await residency(d,pipeline,n,steps,mode);if(i>=warmups)residencyRuns.push({...result,trial:i-warmups+1})}
 const cp=await checkpoint(d,n,chunk);if(i>=warmups)checkpointRuns.push({...cp,trial:i-warmups+1});
 $('summary').textContent='Warmups '+Math.min(i+1,warmups)+'/'+warmups+'; measured '+Math.max(0,i-warmups+1)+'/'+trials;await yieldUI();
 }
 const modes={};for(const mode of ['gpu-resident','readback-each']){const r=residencyRuns.filter(x=>x.mode===mode);modes[mode]={timeMs:stat(r.map(x=>x.elapsedMs)),allValid:r.every(x=>x.valid),intermediateBytesRead:r[0]?.intermediateBytesRead}}
 const cp={snapshotMs:stat(checkpointRuns.map(x=>x.snapshotMs)),restoreMs:stat(checkpointRuns.map(x=>x.restoreMs)),verificationMs:stat(checkpointRuns.map(x=>x.verificationMs)),allValid:checkpointRuns.every(x=>x.valid)};
 last={version:1,createdAt:new Date().toISOString(),gpu:{vendor:adapter.info?.vendor??'',architecture:adapter.info?.architecture??''},configuration:{elements:n,dispatches:steps,trials,warmups,chunkElements:chunk},summary:{residency:modes,checkpoint:cp},residencyRuns,checkpointRuns,notes:['Each GPU-resident step submits a separate command buffer without intermediate readback.','Readback-each waits for every step and copies results to CPU memory.','This measures synchronization cost, not network transport or cross-device state migration.','Checkpoint snapshot uses raw f32 bytes; it does not compress or transmit them.']};
 $('summary').textContent=JSON.stringify(last.summary,null,2);$('details').textContent=JSON.stringify({residencyRuns,checkpointRuns,notes:last.notes},null,2);$('export').disabled=false;
 }finally{d.destroy()}
 }catch(e){$('summary').textContent='Experiment failed: '+String(e)}finally{btn.disabled=false}};
$('export').onclick=()=>{if(!last)return;const url=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='cwgpu-residency-checkpoint-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
