const $ = id => document.getElementById(id);
const state = { device: null, adapter: null, last: null };
const MAP_READ = GPUBufferUsage?.MAP_READ ?? 1;
const STORAGE = GPUBufferUsage?.STORAGE ?? 128;
const COPY_SRC = GPUBufferUsage?.COPY_SRC ?? 4;
const COPY_DST = GPUBufferUsage?.COPY_DST ?? 8;
const shader = instrumented => `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${instrumented ? '@group(0) @binding(3) var<storage, read_write> counter: atomic<u32>;' : ''}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < arrayLength(&output)) {
    output[index] = left[index] + right[index];
    ${instrumented ? 'atomicAdd(&counter, 1u);' : ''}
  }
}`;
function safeNumber(value,min,max){const n=Number(value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(`Value must be an integer between ${min} and ${max}`);return n}
async function init(){
  if(!navigator.gpu)throw Error('WebGPU unavailable. Use a supported browser over HTTPS.');
  if(state.device)return state.device;
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw Error('No WebGPU adapter available.');
  const timestamps=adapter.features.has('timestamp-query');
  const device=await adapter.requestDevice({requiredFeatures:timestamps?['timestamp-query']:[]});
  device.lost.then(info=>{state.device=null;state.adapter=null;$('status').textContent='Device lost: '+info.message});
  state.adapter=adapter;state.device=device;
  return device;
}
function capabilities(){
  const a=state.adapter,d=state.device;
  return {adapter:{vendor:a.info?.vendor||'',architecture:a.info?.architecture||'',device:a.info?.device||'',description:a.info?.description||''},features:[...a.features].sort(),limits:{maxBufferSize:d.limits.maxBufferSize,maxStorageBufferBindingSize:d.limits.maxStorageBufferBindingSize,maxComputeWorkgroupsPerDimension:d.limits.maxComputeWorkgroupsPerDimension,maxComputeInvocationsPerWorkgroup:d.limits.maxComputeInvocationsPerWorkgroup},timestampQueries:d.features.has('timestamp-query'),userAgent:navigator.userAgent};
}
$('inspect').onclick=async()=>{try{$('status').textContent='Inspecting';await init();$('capabilities').textContent=JSON.stringify(capabilities(),null,2);$('status').textContent='Ready'}catch(e){$('status').textContent='Unavailable';$('capabilities').textContent=String(e)}};
function inputs(count){const left=new Float32Array(count),right=new Float32Array(count);for(let i=0;i<count;i++){left[i]=((i*17)%100003)/100003;right[i]=((i*29+7)%100019)/100019}return {left,right}}
async function trial(device,count,instrumented){
  const {left,right}=inputs(count),size=count*4;
  if(size>device.limits.maxStorageBufferBindingSize||size>device.limits.maxBufferSize)throw Error('Buffer exceeds device limits');
  const resources=[];
  const buffer=(bytes,usage)=>{const b=device.createBuffer({size:bytes,usage});resources.push(b);return b};
  const started=performance.now();
  let querySet=null;
  try{
    const a=buffer(size,STORAGE|COPY_DST),b=buffer(size,STORAGE|COPY_DST),out=buffer(size,STORAGE|COPY_SRC),read=buffer(size,MAP_READ|COPY_DST);
    const counter=instrumented?buffer(4,STORAGE|COPY_SRC|COPY_DST):null;
    const counterRead=instrumented?buffer(4,MAP_READ|COPY_DST):null;
    const useTimestamps=device.features.has('timestamp-query');
    const queryRead=useTimestamps?buffer(16,MAP_READ|COPY_DST):null;
    const queryResolve=useTimestamps?buffer(16,GPUBufferUsage.QUERY_RESOLVE|COPY_SRC):null;
    if(useTimestamps)querySet=device.createQuerySet({type:'timestamp',count:2});
    const module=device.createShaderModule({code:shader(instrumented)});
    const diagnostics=await module.getCompilationInfo();
    const errors=diagnostics.messages.filter(m=>m.type==='error');
    if(errors.length)throw Error(errors.map(e=>e.message).join('; '));
    const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
    const compileMs=performance.now()-started;
    device.queue.writeBuffer(a,0,left);device.queue.writeBuffer(b,0,right);
    if(counter)device.queue.writeBuffer(counter,0,new Uint32Array([0]));
    const bindEntries=[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:out}}];
    if(counter)bindEntries.push({binding:3,resource:{buffer:counter}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindEntries});
    const encoder=device.createCommandEncoder();
    const pass=encoder.beginComputePass(useTimestamps?{timestampWrites:{querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
    encoder.copyBufferToBuffer(out,0,read,0,size);
    if(counter)encoder.copyBufferToBuffer(counter,0,counterRead,0,4);
    if(querySet){encoder.resolveQuerySet(querySet,0,2,queryResolve,0);encoder.copyBufferToBuffer(queryResolve,0,queryRead,0,16)}
    const submitStart=performance.now();
    device.queue.submit([encoder.finish()]);
    const submitMs=performance.now()-submitStart;
    const readStart=performance.now();
    await read.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();
    const readbackMs=performance.now()-readStart;
    let invocationCount=null,gpuTimeMs=null;
    if(counterRead){await counterRead.mapAsync(GPUMapMode.READ);invocationCount=new Uint32Array(counterRead.getMappedRange())[0];counterRead.unmap()}
    if(queryRead){await queryRead.mapAsync(GPUMapMode.READ);const ticks=new BigUint64Array(queryRead.getMappedRange());gpuTimeMs=Number(ticks[1]-ticks[0])/1e6;queryRead.unmap()}
    const validationStart=performance.now();
    let mismatches=0;
    for(let i=0;i<count;i++)if(!Number.isFinite(values[i])||values[i]!==Math.fround(left[i]+right[i]))mismatches++;
    const validationMs=performance.now()-validationStart;
    return {mode:instrumented?'instrumented':'baseline',count,valid:mismatches===0,mismatches,invocationCount,compileMs,submitMs,readbackMs,validationMs,gpuTimeMs,totalMs:performance.now()-started,requestedBufferBytes:resources.reduce((s,r)=>s+r.size,0)};
  }finally{querySet?.destroy();for(const r of resources)r.destroy()}
}
$('run').onclick=async()=>{
  const button=$('run');button.disabled=true;$('results').textContent='Running locally on your GPU…';
  try{
    const count=safeNumber($('count').value,1,500000),reps=safeNumber($('reps').value,1,20);
    const device=await init(),runs=[];
    for(let i=0;i<reps;i++){runs.push(await trial(device,count,false));runs.push(await trial(device,count,true));$('results').textContent=`Completed ${i+1}/${reps} repetitions…`;await new Promise(r=>setTimeout(r,0))}
    const mean=(mode,key)=>{const values=runs.filter(r=>r.mode===mode).map(r=>r[key]).filter(Number.isFinite);return values.length?values.reduce((a,b)=>a+b,0)/values.length:null};
    state.last={createdAt:new Date().toISOString(),capabilities:capabilities(),configuration:{count,reps},summary:{baselineMeanTotalMs:mean('baseline','totalMs'),instrumentedMeanTotalMs:mean('instrumented','totalMs'),baselineMeanGpuMs:mean('baseline','gpuTimeMs'),instrumentedMeanGpuMs:mean('instrumented','gpuTimeMs'),allValid:runs.every(r=>r.valid)},runs};
    $('results').textContent=JSON.stringify(state.last,null,2);$('export').disabled=false;
  }catch(e){$('results').textContent='Experiment failed: '+String(e)}finally{button.disabled=false}
};
$('export').onclick=()=>{if(!state.last)return;const blob=new Blob([JSON.stringify(state.last,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='webgpu-lab-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
