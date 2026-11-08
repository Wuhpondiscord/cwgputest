const q=id=>document.getElementById(id);
const R=1,COPY_SRC=4,COPY_DST=8;
const tick=()=>performance.now(),nap=()=>new Promise(r=>setTimeout(r,0));
let last=null;
const err=e=>e instanceof Error?e.name+': '+e.message:String(e);
const round=x=>Math.round(x*1000)/1000;
function stat(x){if(!x.length)return null;const a=x.slice().sort((u,v)=>u-v),n=a.length;return{count:n,median:round(n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2),mean:round(a.reduce((t,v)=>t+v,0)/n),min:round(a[0]),max:round(a.at(-1)),p95:round(a[Math.ceil(.95*n)-1])}}
async function timed(p,limit=6000){let id;try{return await Promise.race([p,new Promise((_,rej)=>{id=setTimeout(()=>rej(new Error('timeout')),limit)})])}finally{clearTimeout(id)}}
function safeUnmap(buf){try{buf.unmap()}catch{}}
function checkBytes(buf,n,expected){const a=new Uint32Array(buf.getMappedRange(0,n*4));for(let i=0;i<n;i++)if(a[i]!==expected[i])throw Error('incorrect readback at '+i)}
async function lifecycle(device){
 const records=[];
 const add=(name,result,detail,t)=>records.push({name,result,detail,elapsedMs:round(tick()-t)});
 // Mapping cancellation without destroying the buffer.
 {
  const t=tick(),b=device.createBuffer({size:16,usage:R|COPY_DST});
  try{const p=b.mapAsync(R,0,16);b.unmap();try{await timed(p);add('Unmap cancels pending mapping','unexpected-acceptance','map resolved after unmap',t)}
  catch(e){add('Unmap cancels pending mapping',/AbortError/.test(err(e))?'pass':'inconclusive',err(e),t)}}
  catch(e){add('Unmap cancels pending mapping','inconclusive',err(e),t)}
  finally{safeUnmap(b);b.destroy()}
 }
 // Duplicate requests while initial mapping is still pending.
 {
  const t=tick(),b=device.createBuffer({size:16,usage:R|COPY_DST});
  try{
   device.pushErrorScope('validation');
   const first=b.mapAsync(R,0,16);
   let secondError=null;try{await timed(b.mapAsync(R,0,16))}catch(e){secondError=err(e)}
   const validation=await timed(device.popErrorScope());
   let firstError=null;try{await timed(first)}catch(e){firstError=err(e)}
   const rejected=!!(secondError||validation);
   add('Duplicate pending map request',rejected&&!firstError?'pass':firstError?'inconclusive':'unexpected-acceptance',
    {secondError,validationError:validation?.message??null,firstError},t);
  }catch(e){add('Duplicate pending map request','inconclusive',err(e),t)}
  finally{safeUnmap(b);b.destroy()}
 }
 // Map, verify, unmap, then map again (legal state transition).
 {
  const t=tick(),b=device.createBuffer({size:16,usage:R|COPY_DST}),s=device.createBuffer({size:16,usage:COPY_SRC|COPY_DST});
  try{
   const values=new Uint32Array([17,29,41,53]);device.queue.writeBuffer(s,0,values);
   for(let i=0;i<2;i++){
    const e=device.createCommandEncoder();e.copyBufferToBuffer(s,0,b,0,16);device.queue.submit([e.finish()]);
    await timed(b.mapAsync(R,0,16));checkBytes(b,4,values);b.unmap();
   }
   add('Map-unmap-remap after successful read','pass','Two verified 16-byte cycles',t);
  }catch(e){add('Map-unmap-remap after successful read','inconclusive',err(e),t)}
  finally{safeUnmap(b);b.destroy();s.destroy()}
 }
 // A destroyed mapped buffer must not be readable through a new mapped range.
 {
  const t=tick(),b=device.createBuffer({size:16,usage:R|COPY_DST});
  try{
   await timed(b.mapAsync(R,0,16));b.destroy();
   let accepted=false,error=null;
   try{b.getMappedRange();accepted=true}catch(e){error=err(e)}
   add('Destroy after completed mapping',accepted?'unexpected-acceptance':'pass',error??'getMappedRange succeeded after destruction',t)
  }catch(e){add('Destroy after completed mapping','inconclusive',err(e),t)}
  finally{b.destroy()}
 }
 // Destroy before starting a mapping; expected API rejection.
 {
  const t=tick(),b=device.createBuffer({size:16,usage:R|COPY_DST});
  b.destroy();device.pushErrorScope('validation');
  let error=null;try{await timed(b.mapAsync(R,0,16))}catch(e){error=err(e)}
  const validation=await timed(device.popErrorScope());
  add('Map after destruction',error||validation?'pass':'unexpected-acceptance',
   {error,validationError:validation?.message??null},t);
 }
 return records;
}
async function readGroup(device,count,mode){
 const bytes=4096,n=bytes/4,source=device.createBuffer({size:bytes,usage:COPY_SRC|COPY_DST}),
 stages=[],t=tick(),expected=new Uint32Array(n);
 for(let i=0;i<n;i++)expected[i]=((i*374761393)^0x5a5a1234)>>>0;
 try{
  device.queue.writeBuffer(source,0,expected);
  for(let i=0;i<count;i++)stages.push(device.createBuffer({size:bytes,usage:R|COPY_DST}));
  if(mode==='batched'){
   const e=device.createCommandEncoder();for(const b of stages)e.copyBufferToBuffer(source,0,b,0,bytes);
   device.queue.submit([e.finish()]);
   await timed(Promise.all(stages.map(b=>b.mapAsync(R,0,bytes))),10000);
   for(const b of stages){checkBytes(b,n,expected);b.unmap()}
  }else{
   for(const b of stages){
    const e=device.createCommandEncoder();e.copyBufferToBuffer(source,0,b,0,bytes);
    device.queue.submit([e.finish()]);
    await timed(b.mapAsync(R,0,bytes));checkBytes(b,n,expected);b.unmap();
   }
  }
  return{count,mode,elapsedMs:round(tick()-t),gpuSubmissions:mode==='batched'?1:count,
   mapCalls:count,bytesRead:count*bytes,valid:true};
 }finally{for(const b of stages){safeUnmap(b);b.destroy()}source.destroy()}
}
async function main(){
 const run=q('runNext');run.disabled=true;q('exportNext').disabled=true;
 const trials=Number(q('nextTrials').value),warmups=Number(q('nextWarmups').value);
 if(!Number.isSafeInteger(trials)||trials<1||trials>20||!Number.isSafeInteger(warmups)||warmups<0||warmups>5){
  q('nextStatus').textContent='Trials must be 1–20 and warmups 0–5';run.disabled=false;return
 }
 last={kind:'dawn-followup',version:2,startedAt:new Date().toISOString(),config:{trials,warmups,
    chunkBytes:4096,counts:[1,4,12,32],order:'alternates by round and count'},
  environment:{userAgent:navigator.userAgent,webgpu:!!navigator.gpu},
  lifecycle:[],performance:[],failures:[],summaries:{},notes:[
   'All work happens in the local browser; WebGPU backend is not identified by adapter metadata.',
   'Per trial includes staging allocation, GPU queue submit, mapping, CPU verification, teardown.',
   'No GPU exhaustion or malformed wire traffic; maximum individual buffer size is 4 KiB in timing tests.',
   'No native fault injection; passing browser validation does not establish memory safety.',
   'Elapsed times include CPU scheduling, JS, GPU and browser wire overhead.'
  ]};
 let device;
 try{
  if(!navigator.gpu)throw Error('WebGPU unavailable');const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw Error('No adapter');
  last.environment.adapter={vendor:adapter.info?.vendor??'',architecture:adapter.info?.architecture??'',device:adapter.info?.device??''};
  device=await adapter.requestDevice();
  device.addEventListener('uncapturederror',e=>{last.uncapturedErrors??=[];last.uncapturedErrors.push(err(e.error))});
  q('nextStatus').textContent='Mapping lifecycle tests';
  last.lifecycle=await lifecycle(device);q('nextResults').textContent=JSON.stringify(last.lifecycle,null,2);
  for(let r=0;r<trials+warmups;r++){
   const counts=r%2?[32,12,4,1]:[1,4,12,32];
   for(const count of counts){
    const modes=(r+count)%2===0?['sequential','batched']:['batched','sequential'];
    for(const mode of modes){
     q('nextStatus').textContent='Readback scaling: '+(r+1)+'/'+(trials+warmups)+' rounds, '+count+' buffers, '+mode;
     try{
      const rec=await readGroup(device,count,mode);
      if(r>=warmups)last.performance.push({...rec,trial:r-warmups+1})
     }catch(e){const failure={count,mode,round:r+1,error:err(e)};last.failures.push(failure);throw Error(JSON.stringify(failure))}
     await nap();
    }
   }
  }
  for(const count of [1,4,12,32]){
   last.summaries[count]={};
   for(const mode of ['sequential','batched']){
    const arr=last.performance.filter(x=>x.count===count&&x.mode===mode&&x.valid);
    last.summaries[count][mode]=stat(arr.map(x=>x.elapsedMs));
   }
   const a=last.summaries[count].sequential?.median,b=last.summaries[count].batched?.median;
   last.summaries[count].medianSpeedup=a!=null&&b>0?round(a/b):null;
  }
  last.lifecycleSummary={pass:last.lifecycle.filter(x=>x.result==='pass').length,
   unexpectedAcceptance:last.lifecycle.filter(x=>x.result==='unexpected-acceptance').length,
   inconclusive:last.lifecycle.filter(x=>x.result==='inconclusive').length};
  q('nextStatus').textContent='Completed '+last.performance.length+' measured performance samples and '+last.lifecycle.length+' lifecycle checks. Export JSON.';
 }catch(e){last.failures.push({stage:'run',error:err(e)});q('nextStatus').textContent='Stopped: '+err(e)+'; partial results exportable.'}
 finally{device?.destroy();last.finishedAt=new Date().toISOString();q('nextResults').textContent=JSON.stringify(last,null,2);q('exportNext').disabled=false;run.disabled=false}
}
q('runNext').onclick=main;
q('exportNext').onclick=()=>{if(!last)return;const url=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='dawn-followup-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)};
