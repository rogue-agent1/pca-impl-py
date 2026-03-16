#!/usr/bin/env python3
"""PCA — principal component analysis via covariance eigendecomposition."""
import math

def pca(X, n_components=2):
    n, d = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n))/n for j in range(d)]
    Xc = [[X[i][j]-means[j] for j in range(d)] for i in range(n)]
    # Covariance matrix
    cov = [[sum(Xc[k][i]*Xc[k][j] for k in range(n))/(n-1) for j in range(d)] for i in range(d)]
    # Power iteration for top eigenvectors
    components = []
    M = [row[:] for row in cov]
    for _ in range(min(n_components, d)):
        v = [1]*d
        for _ in range(200):
            Mv = [sum(M[i][j]*v[j] for j in range(d)) for i in range(d)]
            norm = math.sqrt(sum(x*x for x in Mv))
            if norm < 1e-15: break
            v = [x/norm for x in Mv]
        ev = sum(sum(M[i][j]*v[j] for j in range(d))*v[i] for i in range(d))
        components.append(v)
        for i in range(d):
            for j in range(d): M[i][j] -= ev*v[i]*v[j]
    # Project
    projected = [[sum(Xc[i][j]*components[c][j] for j in range(d)) for c in range(len(components))] for i in range(n)]
    return projected, components

def main():
    X = [[1,2],[3,4],[5,6],[7,8]]
    proj, comp = pca(X, 1)
    print(f"PC1: {[round(c,4) for c in comp[0]]}")
    print(f"Projected: {[round(p[0],4) for p in proj]}")

if __name__ == "__main__": main()
