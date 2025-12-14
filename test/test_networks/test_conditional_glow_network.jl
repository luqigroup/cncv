# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using Test, Random, LinearAlgebra, Flux
using InvertibleNetworks: get_params, clear_grad!, log_likelihood, ∇log_likelihood, LeakyReLUlayer, ResNet, SummarizedNet
using InvertibleNetworks.CUDA: functional

using CNCV

device = functional() ? gpu : cpu
(device == gpu) && println("Testing on GPU");

# Random seed
Random.seed!(3);

# Define network
nx = 32; ny = 32; nz = 32
n_in = 3
n_cond = 3
n_hidden = 4
batchsize = 2
L = 2
K = 2
split_scales = false
N = (nx,ny)

########################################### Test with split_scales = false N = (nx,ny) #########################
# Invertibility

# Network and input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
X = rand(Float32, N..., n_in, batchsize)  |> device
Cond = rand(Float32, N..., n_cond, batchsize)  |> device

Y, Cond, _ = G.forward(X,Cond)
X_ = G.inverse(Y,Cond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

###################################################################################################
# Test logdet_per_batch option


X_batch = rand(Float32, N..., n_in, batchsize) |> device
Cond_batch = rand(Float32, N..., n_cond, batchsize) |> device

# Test with logdet_per_batch=false (default behavior, scalar output)
Y_scalar, Cond_scalar, lgdet_scalar = G.forward(X_batch, Cond_batch; logdet_per_batch=false)
@test lgdet_scalar isa Number
@test size(lgdet_scalar) == ()

# Test with logdet_per_batch=true (per-batch vector output)
Y_batch, Cond_vec, lgdet_vector = G.forward(X_batch, Cond_batch; logdet_per_batch=true)
@test lgdet_vector isa AbstractArray
@test length(lgdet_vector) == batchsize

# Test invertibility with per-batch option
X_rec = G.inverse(Y_batch, Cond_vec)
@test isapprox(norm(X_batch - X_rec)/norm(X_batch), 0f0, atol=1f-5)

# Test accuracy of logdet with logdet_per_batch option
@test isapprox(sum(lgdet_vector), lgdet_scalar * batchsize; atol=1f-1)

###################################################################################################
# Test gradients are set and cleared
G.backward(Y, Y, Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


Random.seed!(3);
# Define network
nx = 32; ny = 32; nz = 32
n_in = 2
n_cond = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2
split_scales = true
N = (nx,ny)

########################################### Test with split_scales = true N = (nx,ny) #########################
# Invertibility

# Network and input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
X = rand(Float32, N..., n_in, batchsize)  |> device
Cond = rand(Float32, N..., n_cond, batchsize)  |> device

Y, Cond, _ = G.forward(X,Cond)
X_ = G.inverse(Y,Cond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

###################################################################################################
# Test logdet_per_batch option with split_scales=true


X_batch = rand(Float32, N..., n_in, batchsize) |> device
Cond_batch = rand(Float32, N..., n_cond, batchsize) |> device

# Test with logdet_per_batch=false (default behavior, scalar output)
Y_scalar, Cond_scalar, lgdet_scalar = G.forward(X_batch, Cond_batch; logdet_per_batch=false)
@test lgdet_scalar isa Number
@test size(lgdet_scalar) == ()

# Test with logdet_per_batch=true (per-batch vector output)
Y_batch, Cond_vec, lgdet_vector = G.forward(X_batch, Cond_batch; logdet_per_batch=true)
@test lgdet_vector isa AbstractArray
@test length(lgdet_vector) == batchsize

# Test invertibility with per-batch option
X_rec = G.inverse(Y_batch, Cond_vec)
@test isapprox(norm(X_batch - X_rec)/norm(X_batch), 0f0, atol=1f-5)

# Test accuracy of logdet with logdet_per_batch option
@test isapprox(sum(lgdet_vector), lgdet_scalar * batchsize; atol=1f-1)

###################################################################################################
# Test gradients are set and cleared
G.backward(Y, Y, Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)

###################################################################################################
# Gradient test

function loss(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = G.backward(ΔY, Y, ZC)
    return f, ΔX, G.CL[1,1].RB.W1.grad, G.CL[1,1].C.v1.grad
end


# Gradient test w.r.t. input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer())  |> device
X = rand(Float32, N..., n_in, batchsize)  |> device
Cond = rand(Float32, N..., n_cond, batchsize)  |> device
X0 = rand(Float32, N..., n_in, batchsize)  |> device
Cond0 = rand(Float32, N..., n_cond, batchsize)  |> device

dX = X - X0

f0, ΔX = loss(G, X0, Cond0)[1:2]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss(G, X0 + h*dX, Cond0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize) |> device
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.CL[1,1].RB.W1.data - G0.CL[1,1].RB.W1.data
dv = G.CL[1,1].C.v1.data - G0.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss(G0, X, Cond)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.CL[1,1].RB.W1.data = Gini.CL[1,1].RB.W1.data + h*dW
    G0.CL[1,1].C.v1.data = Gini.CL[1,1].C.v1.data + h*dv

    f = loss(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)


###################################################################################################
# Gradient test with logdet_per_batch option

# Loss function with logdet_per_batch
function loss_per_batch(G, X, Cond)
    Y, ZC, logdet_vec = G.forward(X, Cond; logdet_per_batch=true)
    batchsize_loc = size(X)[end]
    f = -log_likelihood(Y) - sum(logdet_vec)/batchsize_loc
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = G.backward(ΔY, Y, ZC)
    return f, ΔX, G.CL[1,1].RB.W1.grad, G.CL[1,1].C.v1.grad
end

batchsize_pb = 4
X_pb = rand(Float32, N..., n_in, batchsize_pb) |> device
Cond_pb = rand(Float32, N..., n_cond, batchsize_pb) |> device
X0_pb = rand(Float32, N..., n_in, batchsize_pb) |> device
Cond0_pb = rand(Float32, N..., n_cond, batchsize_pb) |> device
dX_pb = X_pb - X0_pb

G_pb = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N), rb_activation=LeakyReLUlayer()) |> device

# Gradient test w.r.t. input with logdet_per_batch
f0_pb, ΔX_pb = loss_per_batch(G_pb, X0_pb, Cond0_pb)[1:2]
h = 0.1f0
maxiter = 4
err_pb1 = zeros(Float32, maxiter)
err_pb2 = zeros(Float32, maxiter)

print("\nGradient test glow with logdet_per_batch: input\n")
for j=1:maxiter
    f = loss_per_batch(G_pb, X0_pb + h*dX_pb, Cond0_pb)[1]
    err_pb1[j] = abs(f - f0_pb)
    err_pb2[j] = abs(f - f0_pb - h*dot(dX_pb, ΔX_pb))
    print(err_pb1[j], "; ", err_pb2[j], "\n")
    global h = h/2f0
end

@test isapprox(err_pb1[end] / (err_pb1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err_pb2[end] / (err_pb2[1]/4^(maxiter-1)), 1f0; atol=1f0)


########################################### Test with split_scales = true N = (nx,ny) and summary network #########################
# Invertibility
sum_net = ResNet(n_cond, 16, 3; norm=nothing) # make sure it doesnt have any weird normalizations

# Network and input
flow = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer())
G = SummarizedNet(flow, sum_net)  |> device

X = rand(Float32, N..., n_in, batchsize) |> device;
Cond = rand(Float32, N..., n_cond, batchsize) |> device;

Y, ZCond, _ = G.forward(X,Cond)
X_ = G.inverse(Y,ZCond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, ZCond; Y_save = Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2+12) # depends on summary net you use

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test
function loss_sum(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = G.backward(ΔY, Y, ZC; Y_save=Cond)
    return f, ΔX, G.cond_net.CL[1,1].RB.W1.grad, G.cond_net.CL[1,1].C.v1.grad
end

# Gradient test w.r.t. input
X = rand(Float32, N..., n_in, batchsize) |> device;
Cond = rand(Float32, N..., n_cond, batchsize) |> device;
X0 = rand(Float32, N..., n_in, batchsize) |> device;
Cond0 = rand(Float32, N..., n_cond, batchsize) |> device;

dX = X - X0

f0, ΔX = loss_sum(G, X0, Cond0)[1:2]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss_sum(G, X0 + h*dX, Cond0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize) |> device
flow0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
G0 = SummarizedNet(flow0, sum_net) |> device
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.cond_net.CL[1,1].RB.W1.data - G0.cond_net.CL[1,1].RB.W1.data
dv = G.cond_net.CL[1,1].C.v1.data - G0.cond_net.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss_sum(G0, X, Cond)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.cond_net.CL[1,1].RB.W1.data = Gini.cond_net.CL[1,1].RB.W1.data + h*dW
    G0.cond_net.CL[1,1].C.v1.data = Gini.cond_net.CL[1,1].C.v1.data + h*dv

    f = loss_sum(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)


N = (nx,ny,nz)
########################################### Test with split_scales = true N = (nx,ny,nz) #########################
# Invertibility

# Network and input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
X = rand(Float32, N..., n_in, batchsize) |> device
Cond = rand(Float32, N..., n_cond, batchsize) |> device

Y, Cond, _ = G.forward(X,Cond)
X_ = G.inverse(Y,Cond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test


# Gradient test w.r.t. input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
X = rand(Float32, N..., n_in, batchsize) |> device
Cond = rand(Float32, N..., n_cond, batchsize) |> device
X0 = rand(Float32, N..., n_in, batchsize) |> device
Cond0 = rand(Float32, N..., n_cond, batchsize) |> device

dX = X - X0

f0, ΔX = loss(G, X0, Cond0)[1:2]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss(G, X0 + h*dX, Cond0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize) |> device
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.CL[1,1].RB.W1.data - G0.CL[1,1].RB.W1.data
dv = G.CL[1,1].C.v1.data - G0.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss(G0, X, Cond)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.CL[1,1].RB.W1.data = Gini.CL[1,1].RB.W1.data + h*dW
    G0.CL[1,1].C.v1.data = Gini.CL[1,1].C.v1.data + h*dv

    f = loss(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)


########################################### Test with split_scales = true N = (nx,ny,nz) and Summary network #########################
# Invertibility
sum_net_3d = ResNet(n_cond, 16, 3; ndims=3, norm=nothing)  |> device# make sure it doesnt have any weird normalizati8ons

# Network and input
flow = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device;
G = SummarizedNet(flow, sum_net_3d) |> device

X = rand(Float32, N..., n_in, batchsize) |> device;
Cond = rand(Float32, N..., n_cond, batchsize) |> device;

Y, ZCond, _ = G.forward(X,Cond);
X_ = G.inverse(Y,ZCond); # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, ZCond; Y_save=Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2+12)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test


# Gradient test w.r.t. input
X = rand(Float32, N..., n_in, batchsize) |> device;
Cond = rand(Float32, N..., n_cond, batchsize) |> device;
X0 = rand(Float32, N..., n_in, batchsize) |> device;
Cond0 = rand(Float32, N..., n_cond, batchsize) |> device;

dX = X - X0;

f0, ΔX = loss_sum(G, X0, Cond0)[1:2];
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss_sum(G, X0 + h*dX, Cond0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)

# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize) |> device
flow0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N), rb_activation=LeakyReLUlayer()) |> device
G0 = SummarizedNet(flow0, sum_net_3d) |> device
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.cond_net.CL[1,1].RB.W1.data - G0.cond_net.CL[1,1].RB.W1.data
dv = G.cond_net.CL[1,1].C.v1.data - G0.cond_net.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss_sum(G0, X, Cond);
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.cond_net.CL[1,1].RB.W1.data = Gini.cond_net.CL[1,1].RB.W1.data + h*dW
    G0.cond_net.CL[1,1].C.v1.data = Gini.cond_net.CL[1,1].C.v1.data + h*dv

    f = loss_sum(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)