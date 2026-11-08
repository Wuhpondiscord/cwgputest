const $=id=>document.getElementById(id);
let adapter=null,device=null,last=null;
const U={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128,QUERY_RESOLVE:512};
const format=n=>Number.isFinite(n)?Number(n.toFixed(4)):null;
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const median=a=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);return (b[Math.floor((b.length-1)/2)]+b[Math.floor(b.length/2)])/2};
const percentile=(a,p)=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);return b[Math.floor((b.length-1)*p)]};
const sleep=()=>new Promise(r=>setTimeout(r,0));
const clampInt=(id,min,max)=>{const n=Number($(id).value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(id+' must be an integer '+min+'–'+max);return n};
async function init(){
 if(!navigator.gpu)throw Error('WebGPU unavailable. Open via HTTPS in a supported browser.');
 if(device)return device;
 adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No adapter available');
 const features=adapter.features.has('timestamp-query')?['timestamp-query']:[];
 device=await adapter.requestDevice({requiredFeatures:features});
 const current=device;void current.lost.then(info=>{if(device===current){device=null;adapter=null;$('status').textContent='Device lost: '+info.message}});
 return device;
}
function capabilities(){
 return {adapter:{vendor:adapter.info?.vendor??'',architecture:adapter.info?.architecture??'',device:adapter.info?.device??'',description:adapter.info?.description??''},
 features:[...adapter.features].sort(),limits:{maxBufferSize:device.limits.maxBufferSize,maxStorageBufferBindingSize:device.limits.maxStorageBufferBindingSize,maxComputeWorkgroupsPerDimension:device.limits.maxComputeWorkgroupsPerDimension,maxComputeInvocationsPerWorkgroup:device.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupStorageSize:device.limits.maxComputeWorkgroupStorageSize,maxComputeWorkgroupSizeX:device.limits.maxComputeWorkgroupSizeX,maxComputeWorkgroupSizeY:device.limits.maxComputeWorkgroupSizeY,maxComputeWorkgroupSizeZ:device.limits.maxComputeWorkgroupSizeZ,maxBindingsPerBindGroup:device.limits.maxBindingsPerBindGroup},
 navigator:{userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency??null,deviceMemoryGiB:navigator.deviceMemory??null,crossOriginIsolated:crossOriginIsolated,visibility:document.visibilityState},
 timestampQuery:device.features.has('timestamp-query')};
}
$('inspect').onclick=async()=>{try{$('status').textContent='Inspecting';await init();$('capabilities').textContent=JSON.stringify(capabilities(),null,2);$('status').textContent='Ready'}catch(e){$('status').textContent='Unavailable';$('capabilities').textContent=String(e)}};
function inputs(count){const a=new Float32Array(count),b=new Float32Array(count);for(let i=0;i<count;i++){a[i]=((i*17)%100003)/100003;b[i]=((i*29+7)%100019)/100019}return {a,b}}
function wgsl(instrumented,iterations){return `
@group(0) @binding(0) var<storage,read> left: array<f32>;
@group(0) @binding(1) var<storage,read> right: array<f32>;
@group(0) @binding(2) var<storage,read_write> output: array<f32>;
${instrumented?'@group(0) @binding(3) var<storage,read_write> counters: array<atomic<u32>, 2>;':''}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;
 ${instrumented?'atomicAdd(&counters[0],1u);':''}
 if(i<arrayLength(&output)){
  var v=left[i]+right[i];
  ${iterations>0?`for(var j:u32=0u;j<${iterations}u;j++){v=v*1.0000001192092896+0.0000001192092896;}`:''}
  output[i]=v;
  ${instrumented?'atomicAdd(&counters[1],1u);':''}
 }
}`}
async function makePipeline(d,instrumented,iterations){
 const t=performance.now(),module=d.createShaderModule({code:wgsl(instrumented,iterations)});
 const info=await module.getCompilationInfo(),errors=info.messages.filter(x=>x.type==='error');
 if(errors.length)throw Error(errors.map(x=>x.message).join('; '));
 const pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
 return {pipeline,compileMs:performance.now()-t,warnings:info.messages.filter(x=>x.type==='warning').map(x=>x.message)};
}
function expected(a,b,iterations){let v=Math.fround(a+b);for(let j=0;j<iterations;j++)v=Math.fround(Math.fround(v*1.0000001192092896)+0.0000001192092896);return v}
async function execute(d,p,count,iterations,withCounters,withTimestamps){
 const size=count*4,{a,b}=inputs(count),resources=[],start=performance.now();
 const buf=(bytes,usage)=>{const x=d.createBuffer({size:bytes,usage});resources.push(x);return x};
 let qs=null;
 try{
 const left=buf(size,U.STORAGE|U.COPY_DST),right=buf(size,U.STORAGE|U.COPY_DST),output=buf(size,U.STORAGE|U.COPY_SRC),read=buf(size,U.COPY_DST|U.MAP_READ);
 const counters=withCounters?buf(8,U.STORAGE|U.COPY_SRC|U.COPY_DST):null,cr=withCounters?buf(8,U.COPY_DST|U.MAP_READ):null;
 const qr=withTimestamps?buf(16,U.COPY_DST|U.MAP_READ):null,qres=withTimestamps?buf(16,U.QUERY_RESOLVE|U.COPY_SRC):null;
 if(withTimestamps)qs=d.createQuerySet({type:'timestamp',count:2});
 const setupMs=performance.now()-start;
 const uploadStart=performance.now();d.queue.writeBuffer(left,0,a);d.queue.writeBuffer(right,0,b);if(counters)d.queue.writeBuffer(counters,0,new Uint32Array(2));const uploadMs=performance.now()-uploadStart;
 const entries=[{binding:0,resource:{buffer:left}},{binding:1,resource:{buffer:right}},{binding:2,resource:{buffer:output}}];if(counters)entries.push({binding:3,resource:{buffer:counters}});
 const group=d.createBindGroup({layout:p.pipeline.getBindGroupLayout(0),entries});
 const encStart=performance.now(),enc=d.createCommandEncoder();
 const pass=enc.beginComputePass(qs?{timestampWrites:{querySet:qs,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
 pass.setPipeline(p.pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
 enc.copyBufferToBuffer(output,0,read,0,size);if(counters)enc.copyBufferToBuffer(counters,0,cr,0,8);
 if(qs){enc.resolveQuerySet(qs,0,2,qres,0);enc.copyBufferToBuffer(qres,0,qr,0,16)}
 const cmd=enc.finish(),encodeMs=performance.now()-encStart;
 const submitStart=performance.now();d.queue.submit([cmd]);const submitMs=performance.now()-submitStart;
 const readStart=performance.now();await read.mapAsync(1);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();const readbackWaitMs=performance.now()-readStart;
 let invocations=null,validInvocations=null;
 if(cr){await cr.mapAsync(1);const v=new Uint32Array(cr.getMappedRange());invocations=v[0];validInvocations=v[1];cr.unmap()}
 let rawTimestampNs=null,gpuTimeMs=null,timestampStatus=qs?'unavailable':'unsupported';
 if(qr){await qr.mapAsync(1);const v=new BigUint64Array(qr.getMappedRange());rawTimestampNs=[v[0].toString(),v[1].toString()];if(v[1]>v[0]){gpuTimeMs=Number(v[1]-v[0])/1e6;timestampStatus='reported-unverified'}else timestampStatus='zero-or-nonmonotonic';qr.unmap()}
 const vt=performance.now();let mismatches=0,maxAbsError=0,firstMismatch=null;
 for(let i=0;i<count;i++){const ex=expected(a[i],b[i],iterations),actual=values[i],delta=Math.abs(ex-actual);maxAbsError=Math.max(maxAbsError,delta);const tolerance=iterations?Math.max(1e-5,Math.abs(ex)*1e-4):0;if(!Number.isFinite(actual)||delta>tolerance){mismatches++;if(firstMismatch===null)firstMismatch={index:i,expected:ex,actual}}}
 const validationMs=performance.now()-vt;
 return {mode:withCounters?'instrumented':'baseline',valid:mismatches===0,mismatches,firstMismatch,maxAbsError:format(maxAbsError),invocations,validInvocations,expectedInvocations:Math.ceil(count/64)*64,setupMs:format(setupMs),uploadMs:format(uploadMs),encodeMs:format(encodeMs),submitMs:format(submitMs),readbackWaitMs:format(readbackWaitMs),validationMs:format(validationMs),totalMs:format(performance.now()-start),gpuTimeMs:format(gpuTimeMs),rawTimestampNs,timestampStatus,requestedBufferBytes:resources.reduce((s,x)=>s+x.size,0)};
 }finally{qs?.destroy();resources.forEach(x=>x.destroy())}
}
function summarize(runs){
 const result={};for(const mode of ['baseline','instrumented']){
 const rows=runs.filter(r=>r.mode===mode);result[mode]={valid:rows.every(r=>r.valid),sampleCount:rows.length};
 for(const k of ['totalMs','readbackWaitMs','validationMs','gpuTimeMs','uploadMs','encodeMs','setupMs']){
 const values=rows.map(r=>r[k]).filter(x=>typeof x==='number'&&Number.isFinite(x));
 result[mode][k]={median:format(median(values)),mean:format(mean(values)),p95:format(percentile(values,.95)),min:format(Math.min(...values)),max:format(Math.max(...values))};
 if(!values.length){result[mode][k].min=null;result[mode][k].max=null}
 }
 }return result;
}
$('run').onclick=async()=>{
 const button=$('run');button.disabled=true;$('export').disabled=true;$('results').textContent='Preparing experiment…';
 try{
 const count=clampInt('count',1,500000),reps=clampInt('reps',1,50),warmups=clampInt('warmups',0,10),iterations=clampInt('iterations',0,500);
 const d=await init(),bytes=count*4;if(bytes>d.limits.maxStorageBufferBindingSize||bytes>d.limits.maxBufferSize)throw Error('Device buffer limit exceeded');
 const timestamp=d.features.has('timestamp-query'),compiled={};
 for(const mode of ['baseline','instrumented'])compiled[mode]=await makePipeline(d,mode==='instrumented',iterations);
 const runs=[];for(let i=0;i<warmups+reps;i++){
  const order=i%2===0?['baseline','instrumented']:['instrumented','baseline'];
  for(const mode of order){const result=await execute(d,compiled[mode],count,iterations,mode==='instrumented',timestamp);if(i>=warmups)runs.push({...result,round:i-warmups+1});}
  $('results').textContent=`Warmups ${Math.min(i+1,warmups)}/${warmups} · measured ${Math.max(0,i+1-warmups)}/${reps}`;await sleep();
 }
 last={version:2,createdAt:new Date().toISOString(),capabilities:capabilities(),config:{count,reps,warmups,iterations},compilation:{baseline:compiled.baseline.compileMs,instrumented:compiled.instrumented.compileMs},summary:summarize(runs),runs,notes:['GPU timestamps are raw browser-reported values and are not independently calibrated.','CPU wall times include browser scheduling and synchronization.','Arithmetic-loop results use approximate tolerance; vector addition without loop uses exact f32 comparison.','Requested buffer bytes are not physical VRAM measurements.']};
 $('results').textContent=JSON.stringify(last,null,2);$('export').disabled=false;
 }catch(e){$('results').textContent='Experiment failed: '+String(e)}finally{button.disabled=false}
};
$('export').onclick=()=>{if(!last)return;const blob=new Blob([JSON.stringify(last,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='cwgpu-lab-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
