// Standalone control-flow regression harness for two Dawn D3D12 error paths.
// IMPORTANT: Does not link to Dawn or invoke actual ID3D12Resource/ID3D12Fence.
// Mimics the specific lock/map and fence-register error-unwind order so that
// proposed repairs can be checked before porting tests to Dawn's native suite.
// Build: c++ -std=c++17 -O2 -Wall -Wextra -pedantic fault_harness.cpp -o fault_harness
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
struct Failure : std::runtime_error { using runtime_error::runtime_error; };
struct FakeResidency {
    unsigned locks=0;
    bool lockFails=false;
    void lock() { if(lockFails) throw Failure("lock failed"); ++locks; }
    void unlock() { if(!locks) throw Failure("unlock underflow"); --locks; }
};
struct FakeResource {
    bool mapFails=false;
    unsigned mapCalls=0;
    void map() { ++mapCalls; if(mapFails) throw Failure("injected ID3D12Resource::Map failure"); }
};
struct Guard {
    FakeResidency& residency;
    bool committed=false;
    explicit Guard(FakeResidency& r):residency(r){residency.lock();}
    ~Guard(){if(!committed)residency.unlock();}
    void commit(){committed=true;}
};
void observedUnprotectedPath(FakeResidency& r, FakeResource& d){
    // Source-derived control-flow illustration: early error return skips unlock.
    r.lock();
    d.map(); // If this fails the residency lock remains held.
}
void proposedProtectedPath(FakeResidency& r, FakeResource& d){
    Guard g(r);
    d.map();
    g.commit(); // normal subsequent unmap is responsible for unlocking
}
struct FakeFence {
    bool fails=false;
    unsigned calls=0;
    bool registerCompletion() { ++calls; return !fails; }
};
void observedIgnoredFenceStatus(FakeFence& f){ (void)f.registerCompletion(); }
bool proposedCheckedFenceStatus(FakeFence& f){return f.registerCompletion();}
int main(){
    unsigned passed=0,failed=0;
    auto test=[&](const std::string& name,bool condition){
        std::cout<<(condition?"PASS ":"FAIL ")<<name<<"\n";
        (condition?passed:failed)++;
    };
    {
        FakeResidency r;FakeResource d;d.mapFails=true;
        try{observedUnprotectedPath(r,d);}catch(const Failure&){}
        test("Source-shaped Map failure leaves lock (baseline reproduces issue)",r.locks==1&&d.mapCalls==1);
    }
    {
        FakeResidency r;FakeResource d;d.mapFails=true;
        try{proposedProtectedPath(r,d);}catch(const Failure&){}
        test("RAII cleanup balances lock on Map failure",r.locks==0&&d.mapCalls==1);
    }
    {
        FakeResidency r;FakeResource d;proposedProtectedPath(r,d);
        bool wasLocked=r.locks==1;r.unlock();
        test("RAII cleanup preserves normal map/unmap ownership",wasLocked&&r.locks==0);
    }
    {
        FakeResidency r;FakeResource d;r.lockFails=true;
        try{proposedProtectedPath(r,d);}catch(const Failure&){}
        test("Lock failure does not invoke Map or require unlock",r.locks==0&&d.mapCalls==0);
    }
    {
        FakeFence f;f.fails=true;observedIgnoredFenceStatus(f);
        test("Source-shaped fence wrapper discards failure (baseline)",f.calls==1);
    }
    {
        FakeFence f;f.fails=true;
        test("Checked fence wrapper propagates injected failure",!proposedCheckedFenceStatus(f)&&f.calls==1);
    }
    {
        FakeFence f;test("Checked fence wrapper preserves successful completion",proposedCheckedFenceStatus(f)&&f.calls==1);
    }
    std::cout<<"Summary: "<<passed<<" passed, "<<failed<<" failed\n";
    return failed?1:0;
}
