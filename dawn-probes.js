const el=id=>document.getElementById(id);
const MAP_READ=1,COPY_SRC=4,COPY_DST=8,MAP_WRITE=2;
const ms=()=>performance.now();
let report=null;
const sleep=t=>new Promise(r=>setTimeout(r,t));
function errorString(e){return e instanceof Error?e.name+': '+e.message:String(e)}
async function timeout(p,limit=4000){let id;try{return await Promise.race([p,new Promise((_,reject)=>id=setTimeout(()=>reject(new Error('timeout')),limit))])}finally{clearTimeout(id)}}
async function validationProbe(device,name,operation){
  const t=ms();device.pushErrorScope('validation');
  let exception=null,result=null;
  try{result=await timeout(operation())}catch(e){exception=errorString(e)}
  const error=await timeout(device.popErrorScope());
  const rejected=Boolean(exception||error);
  return {name,type:'validation',expected:'Validation error or API rejection',result:rejected?'pass':'unexpected-acceptance',
    elapsedMs:+(ms()-t).toFixed(3),exception,validationError:error?.message??null,returned:result===undefined?null:String(result)};
}
function plain(name,result,detail,start){return{name,type:'lifecycle',expected:'safe bounded operation',result,
  elapsedMs:+(ms()-start).toFixed(3),detail}}
async function probeCopyOutside(device){
 const a=device.createBuffer({size:16,usage:COPY_SRC|COPY_DST}),b=device.createBuffer({size:16,usage:COPY_DST|COPY_SRC});
 try{return await validationProbe(device,'Copy past source buffer end',async()=>{
  const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(a,12,b,0,8);const command=encoder.finish();device.queue.submit([command])
 })}finally{a.destroy();b.destroy()}
}
async function probeMisalignedCopy(device){
 const a=device.createBuffer({size:16,usage:COPY_SRC|COPY_DST}),b=device.createBuffer({size:16,usage:COPY_DST|COPY_SRC});
 try{return await validationProbe(device,'Copy with non-4-byte-aligned offset',async()=>{
  const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(a,2,b,0,4);const command=encoder.finish();device.queue.submit([command])
 })}finally{a.destroy();b.destroy()}
}
async function probeBadMap(device){
 const b=device.createBuffer({size:16,usage:MAP_READ|COPY_DST});
 try{return await validationProbe(device,'MapAsync outside buffer bounds',async()=>await b.mapAsync(MAP_READ,16,8))}
 finally{try{b.unmap()}catch{}b.destroy()}
}
async function probeMisalignedMap(device){
 const b=device.createBuffer({size:16,usage:MAP_READ|COPY_DST});
 try{return await validationProbe(device,'MapAsync invalid offset alignment',async()=>await b.mapAsync(MAP_READ,4,4))}
 finally{try{b.unmap()}catch{}b.destroy()}
}
async function probeDoubleMap(device){
 const b=device.createBuffer({size:16,usage:MAP_READ|COPY_DST});
 const t=ms();try{
  await timeout(b.mapAsync(MAP_READ,0,16));
  const r=await validationProbe(device,'MapAsync while already mapped',async()=>await b.mapAsync(MAP_READ,0,16));
  return r;
 }catch(e){return plain('MapAsync while already mapped','inconclusive',errorString(e),t)}
 finally{try{b.unmap()}catch{}b.destroy()}
}
async function probeDestroyDuringPendingMap(device){
 const t=ms(),b=device.createBuffer({size:16,usage:MAP_READ|COPY_DST});
 let status='inconclusive',detail='';
 try{
  const p=b.mapAsync(MAP_READ,0,16);
  b.destroy();
  try{await timeout(p);status='unexpected-acceptance';detail='mapping succeeded after destruction'}
  catch(e){status=String(e).includes('timeout')?'inconclusive':'pass';detail=errorString(e)}
 }catch(e){status='inconclusive';detail=errorString(e)}finally{b.destroy()}
 return plain('Destroy buffer with pending mapping',status,detail,t);
}
async function probeRoundTrip(device){
 const start=ms(),n=4096,bytes=n*4;
 const source=device.createBuffer({size:bytes,usage:COPY_SRC|COPY_DST});
 const stage=device.createBuffer({size:bytes,usage:COPY_DST|MAP_READ});
 const values=new Uint32Array(n);for(let i=0;i<n;i++)values[i]=i^0x5a5a5a5a;
 try{
  for(let round=0;round<3;round++){
   device.queue.writeBuffer(source,0,values);
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,0,stage,0,bytes);
   device.queue.submit([encoder.finish()]);await timeout(stage.mapAsync(MAP_READ));
   const actual=new Uint32Array(stage.getMappedRange().slice(0));
   if(actual.some((v,i)=>v!==values[i]))return plain('Repeated staging buffer map and unmap','unexpected-acceptance','data mismatch on round '+round,start);
   stage.unmap();
  }
  return plain('Repeated staging buffer map and unmap','pass','3 exact 16-KiB roundtrips',start)
 }catch(e){return plain('Repeated staging buffer map and unmap','inconclusive',errorString(e),start)}
 finally{try{stage.unmap()}catch{}stage.destroy();source.destroy()}
}
async function timingReadback(device,batched){
 const source=device.createBuffer({size:4096,usage:COPY_SRC|COPY_DST});const t=ms(),n=12;
 try{
  device.queue.writeBuffer(source,0,new Uint32Array(1024).fill(1234));
  const stages=Array.from({length:n},()=>device.createBuffer({size:4096,usage:COPY_DST|MAP_READ}));
  try{
   if(batched){
    const enc=device.createCommandEncoder();for(const stage of stages)enc.copyBufferToBuffer(source,0,stage,0,4096);
    device.queue.submit([enc.finish()]);
    await timeout(Promise.all(stages.map(stage=>stage.mapAsync(MAP_READ))),8000);
    for(const stage of stages){if(new Uint32Array(stage.getMappedRange())[0]!==1234)throw Error('readback mismatch');stage.unmap()}
   }else{
    for(const stage of stages){
     const enc=device.createCommandEncoder();enc.copyBufferToBuffer(source,0,stage,0,4096);
     device.queue.submit([enc.finish()]);await timeout(stage.mapAsync(MAP_READ));
     if(new Uint32Array(stage.getMappedRange())[0]!==1234)throw Error('readback mismatch');stage.unmap()
    }
   }
   return plain(batched?'Batched 12 × 4-KiB reads':'Sequential 12 × 4-KiB reads','pass',
     {milliseconds:+(ms()-t).toFixed(3),gpuSubmissions:batched?1:n},t)
  }finally{for(const stage of stages){try{stage.unmap()}catch{}stage.destroy()}}
 }catch(e){return plain(batched?'Batched 12 × 4-KiB reads':'Sequential 12 × 4-KiB reads','inconclusive',errorString(e),t)}
 finally{source.destroy()}
}
async function run(){
 el('run').disabled=true;el('export').disabled=true;
 const records=[],notes=['Untrusted pages cannot directly invoke Dawn C++ helpers or inject D3D12 failures.',
  'A pass establishes expected browser-observable API behavior, not absence of memory corruption.',
  'Unexpected acceptance requires manual verification; it does not establish exploitability.',
  'Runtime uses only bounded buffer allocations (maximum 16 KiB per buffer).'];
 report={kind:'dawn-browser-regression',version:1,createdAt:new Date().toISOString(),
  environment:{userAgent:navigator.userAgent,webgpu:!!navigator.gpu},records,notes};
 let device;
 try{
  if(!navigator.gpu)throw Error('WebGPU is not supported');
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No WebGPU adapter');
  report.environment.adapter={vendor:adapter.info?.vendor??'',architecture:adapter.info?.architecture??'',
    device:adapter.info?.device??'',description:adapter.info?.description??''};
  device=await adapter.requestDevice();
  device.addEventListener('uncapturederror',e=>{report.unhandledErrors??=[];report.unhandledErrors.push(errorString(e.error))});
  const tests=[probeCopyOutside,probeMisalignedCopy,probeBadMap,probeMisalignedMap,
    probeDoubleMap,probeDestroyDuringPendingMap,probeRoundTrip,
    d=>timingReadback(d,false),d=>timingReadback(d,true)];
  for(const test of tests){
   el('status').textContent='Running probe '+(records.length+1)+'/'+tests.length;
   try{records.push(await timeout(test(device),12000))}
   catch(e){records.push({name:test.name,result:'inconclusive',error:errorString(e)})}
   el('results').textContent=JSON.stringify(records,null,2);
   await sleep(0);
  }
  report.summary={pass:records.filter(x=>x.result==='pass').length,
   unexpectedAcceptance:records.filter(x=>x.result==='unexpected-acceptance').length,
   inconclusive:records.filter(x=>x.result==='inconclusive').length};
  el('status').textContent='Completed '+records.length+' bounded probes. Export JSON for review.';
 }catch(e){report.runError=errorString(e);el('status').textContent='Stopped: '+report.runError}
 finally{device?.destroy();report.finishedAt=new Date().toISOString();el('export').disabled=false;el('run').disabled=false;
  el('results').textContent=JSON.stringify(report,null,2)}
}
el('run').onclick=run;
el('export').onclick=()=>{if(!report)return;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='dawn-browser-probes-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)};
