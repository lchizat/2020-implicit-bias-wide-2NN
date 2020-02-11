using PyPlot
using LinearAlgebra, Random


"""
Gradient ascent to train a 2-layers ReLU neural net for the soft-min loss
INPUT: X (training input), Y (training output), m (nb neurons), both: training both layers or just the output
OUTPUT: Ws (training trajectory)
"""
function twonet(X, Y, m, stepsize, niter; both=true) 
    (n,d) = size(X)
    W_init = randn(m, d+1)
    if !both
        W_init[:,end] .= 0
    end

    W     = copy(W_init)
    Ws    = zeros(m, d+1, niter) # store optimization path
    loss  = zeros(niter)
    margins = zeros(niter)
    betas = zeros(niter)

    for iter = 1:niter
        Ws[:,:,iter] = W
        act  =  max.( W[:,1:end-1] * X', 0.0) # (size m × n)
        out  =  (1/m) * sum( W[:,end] .* act , dims=1) # (size 1 × n)
        perf = Y .* out[:]
        margin = minimum(perf)
        temp = exp.(margin .- perf) # stabilization
        gradR = temp .* Y ./ sum(temp)' # size n
        grad_w1 = (W[:,end] .* float.(act .> 0) * ( X .* gradR  ))  # (size m × d) 
        grad_w2 = act * gradR  # size m
        
        if both
            grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
            betas[iter] = sum(W.^2)/m
            loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
            margins[iter] = margin/betas[iter]
            W = W + stepsize * grad/(sqrt(iter+1))
            
         else 
            grad = cat(zeros(m,d), grad_w2, dims=2) # size (m × d+1)
            betas[iter] = maximum([1,sqrt(sum(W[:,end].^2)/m)])
            loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
            margins[iter] = margin/(sqrt(sum(W[:,end].^2))/m)
            W = W + betas[iter] * stepsize * grad /(sqrt(iter+1))
        end
    end
    Ws, loss, margins, betas
end


"Coordinates of the 2d cluster centers, p is k^2 the number of clusters"
function cluster_center(p,k)
    p1 = mod.(p .- 1,k) .+ 1
    p2 = div.(p .- 1,k) .+ 1
    Delta = 1/(3k-1)
    x1 =  Delta*(1 .+ 3*(p1 .- 1)) .- 1/2
    x2 =  Delta*(1 .+ 3*(p2 .- 1)) .- 1/2
    return x1,x2
end

"""
Plot the classifier for a test case, comparing training both or output layer
"""
function illustration(k, n, m; stepsize= 0.5, niter=100000, name="decision")
# data distribution
sd = 0 # number of spurious dimensions
Delta = 1/(3k-1) # interclass distance
A = ones(k^2) # cluster affectation
A[randperm(k^2)[1:div(k^2,2)]] .= -1

# sample from it
P = rand(1:k^2,n) # cluster label
T = 2π*rand(n)  # shift angle
R = Delta*rand(n) # shift magnitude
X = cat(ones(n), cluster_center(P,k)[1] .+ R .* cos.(T),cluster_center(P,k)[2] + R .* sin.(T), (rand(n,sd) .- 1/2), dims=2)
Y = A[P]

# train neural network
Ws1, loss1, margins1, betas1 = twonet(X, Y, m, stepsize, niter; both=true)
Ws2, loss2, margins2, betas2 = twonet(X, Y, m, stepsize, niter; both=false)
    
# plots
X1 = X[(Y .== 1),:]
X2 = X[(Y .== -1),:]
    
figure(figsize=[2.5,2.5])
    plot(X1[:,2],X1[:,3],"+r")
    plot(X2[:,2],X2[:,3],"_b")
    plot(cluster_center(1:k^2,k)[1],cluster_center(1:k^2,k)[2],"ok")
    axis("equal");#axis("off")
    xticks([], []); yticks([], [])
    savefig(name * "setting.pdf",bbox_inches="tight")
    
figure(figsize=[2.5,2.5])
    f1(x1,x2,t) = (1/m) * sum( Ws1[:,end,t] .* max.( Ws1[:,1:3,t] * [1;x1;x2], 0.0)) # (size 1 × n)
    xs = -0.8:0.01:0.8
    tab1 = [f1(xs[i],xs[j],size(Ws1,3)) for i=1:length(xs), j=1:length(xs)]
    pcolormesh(xs', xs, tanh.(1000*tab1'),cmap="coolwarm",shading="gouraud",vmin=-1.0,vmax=1.0,edgecolor="face")
    xs = -0.8:0.005:0.8
    tab1 = [f1(xs[i],xs[j],size(Ws1,3)) for i=1:length(xs), j=1:length(xs)]
    contour(xs', xs, tanh.(1000*tab1'),levels =0, colors="k",antialiased = true,linewidths=2)
    plot(X1[:,2],X1[:,3],"+k")
    plot(X2[:,2],X2[:,3],"_k")
    axis("equal");axis("off");
    savefig(name * "_both.pdf",bbox_inches="tight")
    
 figure(figsize=[2.5,2.5])
    f2(x1,x2,t) = (1/m) * sum( Ws2[:,end,t] .* max.( Ws2[:,1:3,t] * [1;x1;x2], 0.0)) # (size 1 × n)
    xs = -0.8:0.01:0.8
    tab2 = [f2(xs[i],xs[j],size(Ws2,3)) for i=1:length(xs), j=1:length(xs)]
    pcolormesh(xs', xs, tanh.(1000*tab2'),cmap="coolwarm",shading="gouraud",vmin=-1.0,vmax=1.0,edgecolor="face")
    xs = -0.8:0.005:0.8
    tab2 = [f2(xs[i],xs[j],size(Ws2,3)) for i=1:length(xs), j=1:length(xs)]
    contour(xs', xs, tanh.(1000*tab2'),levels =0, colors="k",antialiased = true,linewidths=2)
    plot(X1[:,2],X1[:,3],"+k")
    plot(X2[:,2],X2[:,3],"_k")
    axis("equal");axis("off");
    savefig(name *"_output.pdf",bbox_inches="tight")
end


"""
Compute margin and generalisation error.
Run the experiment with the given list of d (input dimension), n (input points), m (width)
On n_distrib random input distributions and n_repeat random initialization.
"""
function experiment_2NN(n_distrib, ds, ns, n_repeat, ms; k = 3, niter = 5*10^5, stepsize = 1, n_test=4000)

    L1s = zeros(n_distrib, length(ds), length(ns), n_repeat, length(ms))
    L2s = zeros(n_distrib, length(ds), length(ns), n_repeat, length(ms))
    m1s = zeros(n_distrib, length(ds), length(ns), n_repeat, length(ms))
    m2s = zeros(n_distrib, length(ds), length(ns), n_repeat, length(ms))
    
    p = Progress(length(L1s))
    for i=1:n_distrib
        # define the distribution
        Delta = 1/(3k-1) # lower bound on the interclass distance
        A = ones(k^2) 
        A[randperm(k^2)[1:div(k^2,2)]] .= -1 # cluster affectation
        
        for ii = 1: length(ds)
            d = ds[ii]
            # sample from the distribution (we then take a subset of it)
            nmax = maximum(ns)
            P = rand(1:k^2,nmax) # cluster label
            T = 2π*rand(nmax)  # shift angle
            R = Delta*rand(nmax) # shift magnitude
            sd = d-3 # number of spurious dimensions
            X = cat(ones(nmax),
                      cluster_center(P,k)[1] .+ R .* cos.(T), 
                      cluster_center(P,k)[2] + R .* sin.(T), 
                      rand(nmax,sd) .- 1/2,
                      dims=2)
            Y = A[P]
            for iii = 1:length(ns)
                n = ns[iii]
                for iv=1:n_repeat # same eexperiment but with another random init
                    for v=1:length(ms)
                        m = ms[v]
                        
                        # train neural network
                        Ws1, loss1, margins1, betas1 = twonet(X[1:n,:], Y[1:n], m, stepsize, niter; both=true)
                        Ws2, loss2, margins2, betas2 = twonet(X[1:n,:], Y[1:n], m, stepsize, niter; both=false)
    
                        # test set
                        T_test = 2π*rand(n_test)  # shift angle
                        R_test = Delta*rand(n_test) # shift magnitude
                        P_test = rand(1:k^2,n_test) # cluster label
                        X_test = cat(ones(n_test), cluster_center(P_test,k)[1] .+ R_test .* cos.(T_test), 
                        cluster_center(P_test,k)[2] + R_test .* sin.(T_test), (rand(n_test,sd) .- 1/2), dims=2)
                        Y_test = A[P_test]
    
                        # test error
                        preds1 = (1/m) * sum(Ws1[:,end,end] .* max.( Ws1[:,1:end-1,end] * X_test', 0.0), dims = 1) 
                        L1 = sum((Y_test .* preds1') .< 0)/n_test # probability of error
                        preds2 = (1/m) * sum(Ws2[:,end,end] .* max.( Ws2[:,1:end-1,end] * X_test', 0.0), dims = 1) 
                        L2 = sum((Y_test .* preds2') .< 0)/n_test # probability of error
                        L1s[i,ii,iii,iv,v] = L1
                        L2s[i,ii,iii,iv,v] = L2
                        m1s[i,ii,iii,iv,v] = maximum(margins1)
                        m2s[i,ii,iii,iv,v] = maximum(margins2)
                        GC.gc() # garbage collection
                        next!(p)
                    end
                end
            end
        end
    end
    return L1s, L2s, m1s, m2s
end