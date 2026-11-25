// Exact batched finite-PMF propagation. No sampling, pruning or threads.
#include <algorithm>
#include <cstdint>
#include <vector>
#include <memory>
#include <list>
#include <unordered_map>
#include <string>
#include <deque>
#include <array>

namespace {
struct Edge { int to; double probability; };
struct Operator { std::vector<int> offsets; std::vector<Edge> edges; };
// Thread-local PMF/capacity identity prevents cross-case or cross-thread reuse.
// The LRU bounds native storage independently of Python's projection caches.
struct OperatorCache {
    int cap=-1;
    std::vector<int32_t> demands;
    std::vector<double> probabilities;
    std::unordered_map<int, std::pair<std::shared_ptr<Operator>,std::list<int>::iterator>> entries;
    std::list<int> lru;
    size_t bytes=0;
    void bind(int c,const int32_t* d,const double* p,int support) {
        if(c==cap && demands.size()==static_cast<size_t>(support)
            && std::equal(demands.begin(),demands.end(),d)
            && std::equal(probabilities.begin(),probabilities.end(),p)) return;
        entries.clear(); lru.clear(); bytes=0; cap=c;
        demands.assign(d,d+support); probabilities.assign(p,p+support);
    }
    static size_t memory(const Operator& op) {
        return op.offsets.capacity()*sizeof(int)+op.edges.capacity()*sizeof(Edge)+128;
    }
    std::shared_ptr<Operator> get(int ceiling,int arrival) {
        const int base=cap+1, size=base*base, key=ceiling*base+arrival;
        auto found=entries.find(key);
        if(found!=entries.end()) {
            lru.splice(lru.end(),lru,found->second.second);
            return found->second.first;
        }
        auto op=std::make_shared<Operator>();
        op->offsets.reserve(size+1);
        for(int old=0;old<=cap;++old) for(int inventory=0;inventory<=cap;++inventory) {
            op->offsets.push_back(op->edges.size());
            if(inventory<old) continue;
            int previous=-1;
            for(size_t d=0;d<demands.size();++d) {
                const int left=std::max(old-demands[d],0);
                const int expiry=std::max(left-ceiling,0);
                const int o=left-expiry;
                const int i=std::min(std::max(inventory+arrival-demands[d],0)-expiry,cap);
                const int target=o*base+i;
                if(target==previous) op->edges.back().probability+=probabilities[d];
                else { op->edges.push_back({target,probabilities[d]}); previous=target; }
            }
        }
        op->offsets.push_back(op->edges.size());
        const size_t used=memory(*op), limit=32*1024*1024;
        while(bytes+used>limit && !lru.empty()) {
            int victim=lru.front(); lru.pop_front();
            bytes-=memory(*entries.at(victim).first); entries.erase(victim);
        }
        if(used<=limit) {
            lru.push_back(key);
            entries.emplace(key,std::make_pair(op,std::prev(lru.end()))); bytes+=used;
        }
        return op;
    }
};
thread_local OperatorCache operator_cache;
}

extern "C" int ltp_inventory_forward(const int32_t* roots,int count,int width,
    int shelf,int periods,int cap,const int32_t* demands,const double* probabilities,
    int support,double* output) {
    if(count<0 || cap<0 || cap>100 || periods<0 || periods>=shelf) return 1;
    operator_cache.bind(cap,demands,probabilities,support);
    const int base=cap+1, size=base*base;
    std::vector<double> law(size,0.0), next(size,0.0);
    std::vector<int> active, following, marks(size,-1);
    int generation=0;
    for(int r=0;r<count;++r) {
        int total=0; for(int a=0;a<shelf;++a) total+=roots[r*width+a];
        const int initial=total*base+total;
        active.assign(1,initial); law[initial]=1.0;
        int ceiling=total;
        for(int t=0;t<periods;++t) {
            ceiling-=roots[r*width+t];
            auto op=operator_cache.get(ceiling,roots[r*width+shelf+t]);
            following.clear(); ++generation;
            for(int source:active) {
                const double mass=law[source];
                if(mass!=0.0) for(int e=op->offsets[source];e<op->offsets[source+1];++e) {
                    const Edge& edge=op->edges[e];
                    if(marks[edge.to]!=generation) {
                        marks[edge.to]=generation; following.push_back(edge.to);
                    }
                    next[edge.to]+=mass*edge.probability;
                }
                law[source]=0.0;
            }
            law.swap(next); active.swap(following);
        }
        double value=0.0;
        for(int source:active) { value+=law[source]*(source%base); law[source]=0.0; }
        output[2*r]=value; output[2*r+1]=0.0;
    }
    return 0;
}

extern "C" int ltp_reduced_batch(const int32_t* roots, int count, int width,
    int shelf, int periods, int cap, const int32_t* demands,
    const double* probabilities, int support, int want_waste, int want_inventory, double* output) {
    // All roots advance one projection period together. Each root carries a
    // joint law over removed initial stock and total stock; demand branches
    // are accumulated in compiled arithmetic, not Python scalar trajectories.
    if (count < 0 || cap < 0 || cap > 100 || periods > shelf || periods < 0
        || (want_inventory && periods == shelf))
        return 1;
    const int base = cap + 1, size = base * base;
    std::vector<double> tails(support+1,0.0);
    for(int d=support-1;d>=0;--d) tails[d]=tails[d+1]+probabilities[d];
    if (!want_inventory && periods == shelf) {
        // Only the first scheduled receipt can expire within this window.
        // Track its possible first-period capacity discard along with removed
        // stock. Subsequent receipts are younger and cannot affect this law.
        const int wide = 2*cap+1;
        std::vector<double> law(base*wide), next(base*wide);
        for (int r=0; r<count; ++r) {
            std::fill(law.begin(),law.end(),0.0);
            int n=roots[r*width+shelf], deadline=roots[r*width];
            for (int a=0;a<shelf;++a) n+=roots[r*width+a];
            output[2*r]=output[2*r+1]=0.0;
            const int max_discard=std::max(n-(shelf==1?n:deadline)-cap,0);
            for (int d=0;d<support;++d) {
                const int consumed=std::min(n,demands[d]);
                // At shelf=1 the first receipt also expires immediately.
                const int expired=std::max((shelf==1?n:deadline)-consumed,0);
                const int discard=std::max(n-consumed-expired-cap,0);
                if(consumed+expired<n-discard)
                    law[discard*wide+consumed+expired]+=probabilities[d];
                if(want_waste) output[2*r+1]+=probabilities[d]*expired;
            }
            for(int t=1;t<periods;++t) {
                deadline+=roots[r*width+t];
                std::fill(next.begin(),next.end(),0.0);
                for(int discard=0;discard<=max_discard;++discard) {
                    const int effective=n-discard;
                    if(effective<0) continue;
                    const int boundary=(t==shelf-1?effective:deadline);
                    for(int w=0;w<effective;++w) {
                        const double mass=law[discard*wide+w];
                        if(mass==0.0) continue;
                        for(int d=0;d<support;++d) {
                            // Exhausted tracked stock never produces further
                            // expiry, so its complete PMF tail needs no state.
                            if(demands[d]>=effective-w) break;
                            const int consumed=std::min(effective,w+demands[d]);
                            const int expired=std::max(boundary-consumed,0);
                            const double weight=mass*probabilities[d];
                            if(consumed+expired<effective)
                                next[discard*wide+consumed+expired]+=weight;
                            if(want_waste) output[2*r+1]+=weight*expired;
                        }
                    }
                }
                law.swap(next);
            }
        }
        return 0;
    }
    if (!want_inventory) {
        std::vector<double> law(static_cast<size_t>(count)*base, 0.0), next(law.size());
        std::vector<int> initial(count, 0), deadline(count, 0);
        for (int r=0; r<count; ++r) {
            for (int a=0; a<shelf; ++a) initial[r] += roots[r*width+a];
            law[r*base] = 1.0;
            output[2*r] = output[2*r+1] = 0.0;
        }
        for (int t=0; t<periods; ++t) {
            std::fill(next.begin(), next.end(), 0.0);
            for (int r=0; r<count; ++r) {
                deadline[r] += roots[r*width+t];
                for (int w=0; w<initial[r]; ++w) {
                    const double mass=law[r*base+w];
                    if (mass==0.0) continue;
                    for (int d=0; d<support; ++d) {
                        if(demands[d]>=initial[r]-w) break;
                        const int consumed=std::min(initial[r],w+demands[d]);
                        const int expired=std::max(deadline[r]-consumed,0);
                        const double weight=mass*probabilities[d];
                        if(consumed+expired<initial[r])
                            next[r*base+consumed+expired] += weight;
                        if (want_waste) output[2*r+1] += weight*expired;
                    }
                }
            }
            law.swap(next);
        }
        return 0;
    }
    std::vector<double> law(static_cast<size_t>(count) * size, 0.0);
    std::vector<double> next(law.size(), 0.0);
    std::vector<int> initial(count, 0), deadline(count, 0);
    for (int r = 0; r < count; ++r) {
        for (int age = 0; age < shelf; ++age) initial[r] += roots[r*width+age];
        law[static_cast<size_t>(r)*size + initial[r]] = 1.0;
        output[2*r] = output[2*r+1] = 0.0;
    }
    for (int t = 0; t < periods; ++t) {
        std::fill(next.begin(), next.end(), 0.0);
        for (int r = 0; r < count; ++r) {
            deadline[r] += roots[r*width+t];
            const int n = initial[r], e = deadline[r];
            const int arrival = roots[r*width+shelf+t];
            const size_t offset = static_cast<size_t>(r)*size;
            for (int removed = 0; removed <= n; ++removed) {
                for (int inventory = n-removed; inventory <= cap; ++inventory) {
                    const double mass = law[offset + removed*base + inventory];
                    if (mass == 0.0) continue; // exact zero, never threshold pruning
                    for (int d = 0; d < support; ++d) {
                        if(demands[d]>=std::max(n-removed,inventory+arrival)) {
                            next[offset+n*base] += mass*tails[d];
                            break;
                        }
                        const int consumed = std::min(n, removed + demands[d]);
                        const int expired = std::max(e-consumed, 0);
                        const int w = consumed + expired;
                        const int i = std::min(std::max(inventory+arrival-demands[d], 0)-expired, cap);
                        const double weight = mass * probabilities[d];
                        next[offset + w*base+i] += weight;
                        if (want_waste) output[2*r+1] += weight * expired;
                    }
                }
            }
        }
        law.swap(next);
    }
    for (int r = 0; r < count; ++r)
        for (int w = 0; w <= initial[r]; ++w)
            for (int i = initial[r]-w; i <= cap; ++i)
                output[2*r] += law[static_cast<size_t>(r)*size+w*base+i] * i;
    return 0;
}

namespace {
// Each Python projector owns this context: distinct PMFs/quantities cannot
// evict or accidentally reuse one another's expectations. All branches remain
// exact; only identical complete states are merged. Bound storage per depth.
struct LongProjector {
    using Value = std::array<double, 2>;
    struct Layer {
        std::unordered_map<std::string, Value> values;
        std::deque<std::string> fifo;
    };
    int shelf, lead, cap, quantity, limit;
    bool expiry, cached_inventory;
    std::vector<int32_t> demands;
    std::vector<double> probabilities;
    std::vector<Layer> layers;
    uint64_t expanded=0, hits=0, branches=0, leaves=0;

    Value solve(std::vector<int32_t> state, int remaining) {
        if (remaining == 0) {
            int stock=0; for(int a=0;a<shelf;++a) stock+=state[a];
            return {{static_cast<double>(stock),0.0}};
        }
        if(quantity==1) {
            // Only receipts arriving early enough to expire can contribute
            // waste. Younger receipts cannot consume or displace older stock:
            // service is FIFO and capacity discards the youngest stock first.
            // Canonicalize before lookup, at EVERY depth, not just the tail.
            int relevant=std::max(0,remaining-shelf+1);
            for(int i=shelf+relevant;i<shelf+lead;++i) state[i]=0;
        }
        std::string key(state.size(),'\0');
        for(size_t i=0;i<state.size();++i) key[i]=static_cast<char>(state[i]);
        auto& layer=layers[remaining];
        auto found=layer.values.find(key);
        if(found!=layer.values.end()) { ++hits; return found->second; }
        ++expanded;
        Value result={{0.0,0.0}};
        if(remaining<shelf || (remaining==shelf && quantity==1)) {
            ++leaves;
            if(quantity==0 && cached_inventory)
                ltp_inventory_forward(state.data(),1,state.size(),shelf,remaining,cap,
                    demands.data(),probabilities.data(),demands.size(),result.data());
            else
                ltp_reduced_batch(state.data(),1,state.size(),shelf,remaining,cap,
                    demands.data(),probabilities.data(),demands.size(),
                    expiry && quantity!=0,quantity!=1,result.data());
        } else {
            // Demand order matches the Python projector. Equal child states
            // reuse their values but each PMF term is accumulated separately.
            int available=state[shelf];
            for(int a=0;a<shelf;++a) available+=state[a];
            Value last={{0.0,0.0}};
            for(size_t d=0;d<demands.size();++d) {
                if(d>0 && demands[d-1]>=available) {
                    // Every remaining demand empties the stock identically.
                    // Reuse its expectation, retaining each original addition
                    // (including sub-probability PMFs) in the same order.
                    ++branches;
                    result[0]+=last[0]*probabilities[d];
                    result[1]+=last[1]*probabilities[d];
                    continue;
                }
                auto child=state;
                child[shelf-1]+=state[shelf];
                int need=demands[d];
                for(int a=0;a<shelf;++a) {
                    int used=std::min(child[a],need); child[a]-=used; need-=used;
                }
                int waste=child[0], total=0;
                for(int a=0;a<shelf-1;++a) { child[a]=child[a+1]; total+=child[a]; }
                child[shelf-1]=0;
                int discard=std::max(total-cap,0);
                for(int a=shelf-1;a>=0;--a) {
                    int used=std::min(child[a],discard); child[a]-=used; discard-=used;
                }
                for(int i=shelf;i<shelf+lead-1;++i) child[i]=child[i+1];
                child.back()=0;
                ++branches;
                Value next=solve(std::move(child),remaining-1);
                last=next;
                result[0]+=next[0]*probabilities[d];
                result[1]+=(next[1]+(expiry?waste:0))*probabilities[d];
            }
        }
        if(static_cast<int>(layer.values.size())>=limit) {
            layer.values.erase(layer.fifo.front()); layer.fifo.pop_front();
        }
        layer.values.emplace(key,result); layer.fifo.push_back(std::move(key));
        return result;
    }
};
}

extern "C" void* ltp_long_create(int shelf,int lead,int cap,int quantity,int expiry,
    int cached_inventory,int limit,const int32_t* demands,const double* probabilities,int support) {
    if(shelf<1 || lead<1 || cap<0 || cap>100 || support<1 || limit<1) return nullptr;
    auto p=new LongProjector();
    p->shelf=shelf; p->lead=lead; p->cap=cap; p->quantity=quantity;
    p->expiry=expiry; p->cached_inventory=cached_inventory; p->limit=limit;
    p->demands.assign(demands,demands+support);
    p->probabilities.assign(probabilities,probabilities+support);
    p->layers.resize(lead+1);
    return p;
}
extern "C" void ltp_long_destroy(void* context) { delete static_cast<LongProjector*>(context); }
extern "C" int ltp_long_batch(void* context,const int32_t* roots,int count,double* output,uint64_t* stats) {
    if(!context || count<0) return 1;
    auto& p=*static_cast<LongProjector*>(context);
    int width=p.shelf+p.lead;
    for(int r=0;r<count;++r) {
        auto value=p.solve(std::vector<int32_t>(roots+r*width,roots+(r+1)*width),p.lead);
        output[2*r]=value[0]; output[2*r+1]=value[1];
    }
    stats[0]=p.expanded; stats[1]=p.hits; stats[2]=p.branches; stats[3]=p.leaves;
    return 0;
}
