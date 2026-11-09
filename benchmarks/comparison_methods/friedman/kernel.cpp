#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Positional six-mer kernel, following barakcohenlab/preimage v1.0.0.
struct Key {
    int a, b, shift;
    bool operator==(const Key& k) const {
        return a == k.a && b == k.b && shift == k.shift;
    }
};
struct Hash {
    size_t operator()(const Key& k) const {
        return (size_t(k.a) * 1000000007ULL) ^
               (size_t(k.b) * 1000000009ULL) ^ size_t(k.shift + 4);
    }
};
struct Kernel {
    const int32_t* ids;
    const uint16_t* tiles;
    int n, nt, width;
    uint16_t rc[4096];
    double weights[641];
    std::unordered_map<Key, double, Hash> cache;
    Kernel(const int32_t* i, const uint16_t* t, int nn, int tt, int ww)
        : ids(i), tiles(t), n(nn), nt(tt), width(ww) {
        for (int c = 0; c < 4096; ++c) {
            int q = c, rev = 0;
            for (int j = 0; j < 6; ++j) { rev = (rev << 2) | (3 - (q & 3)); q >>= 2; }
            rc[c] = rev;
        }
        for (int d = 0; d <= 640; ++d) weights[d] = std::exp(-double(d*d)/200.0);
    }
    double tile_pair(int a, int b, int shift) {
        if (a > b) { std::swap(a, b); shift = -shift; }
        Key key{a,b,shift};
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        double v = 0;
        const uint16_t* x = tiles + size_t(a) * width;
        const uint16_t* y = tiles + size_t(b) * width;
        for (int i = 0; i < width && x[i] != 65535; ++i) {
            for (int j = 0; j < width && y[j] != 65535; ++j) {
                const int d = std::abs(shift * width + i - j);
                if (d > 386) continue;
                int sim = int(x[i] == y[j]) + int(x[i] == rc[y[j]]);
                if (sim) v += weights[d] * sim;
            }
        }
        if (cache.size() >= 4000000) cache.clear();
        cache.emplace(key, v);
        return v;
    }
    double pair(int a, int b) {
        double total = 0;
        for (int p = 0; p < nt; ++p) {
            int x = ids[size_t(a)*nt+p];
            if (x < 0) break;
            for (int q = std::max(0,p-4); q < std::min(nt,p+5); ++q) {
                int y = ids[size_t(b)*nt+q];
                if (y >= 0) total += tile_pair(x, y, p-q);
            }
        }
        return total;
    }
};
extern "C" {
void* gs_create(const int32_t* ids, const uint16_t* tiles, int n, int nt, int width) {
    return new Kernel(ids,tiles,n,nt,width);
}
void gs_destroy(void* ptr) { delete static_cast<Kernel*>(ptr); }
void gs_diag(void* ptr, double* out) {
    auto* k = static_cast<Kernel*>(ptr);
    for (int i=0; i<k->n; ++i) out[i]=k->pair(i,i);
}
void gs_gram(void* ptr, const int32_t* rows, int nr, const int32_t* cols, int nc, double* out) {
    auto* k = static_cast<Kernel*>(ptr);
    for (int i=0; i<nr; ++i) for (int j=0; j<nc; ++j)
        out[size_t(i)*nc+j]=k->pair(rows[i],cols[j]);
}
}
