import Pkg
# Pkg.add("LinearAlgebra")
# Pkg.add("Plots")
# Pkg.add("Flux")
# Pkg.add("CUDA")
# Pkg.add("MLDatasets")
# Pkg.add("ProgressBars")
# Pkg.add("Images")
# Pkg.add("JuliaInterpreter")
# Pkg.add("MethodAnalysis")
# Pkg.add("GLM")

include("Ecc.jl")
using LinearAlgebra, Plots, MLDatasets, GLM, Colors

images, labels = MNIST.traindata()
labels .+= 1
images = images .> 0.8
images = reshape(images, (1, size(images)...))
split = 40000
trainset = @view images[:,:,:,begin:split]
trainlabels = labels[begin:split]
testset = @view images[:,:,:,split+1:end]
testlabels = labels[split+1:end]

h, w = 4, 5
ph, pw = 5, 5
# sparse_testset = [sparse(x) for x in eachslice(testset, dims=4)]
ecc = layer.hard_wta_l2(ph*pw, h*w)

cs = conv.shape((1,28,28),ecc.m,(5,5))
for epoch ∈ 1:2
    train_results = batch_run_sparse_conv(ecc, cs, trainset)
    classifier_head = head.naive_bayes(conv.out_volume(cs), 10, train_results, trainlabels)
    test_results = batch_run_sparse_conv(ecc, cs, testset)
    pred_testlabels = head.batch_run(classifier_head, test_results)
    accuracy = sum(testlabels .== pred_testlabels) / length(testlabels)
    println("[$(epoch)] accuracy=$(accuracy)")
    train_on_patches(ecc,trainset,split,3,ph,pw)
end


# exit()
# images = images .> 0.8
# width, height, dataset_size = size(images)
# patch = 5
# input_size = patch * patch
# output_size = 25
# W = rand(input_size, output_size)

# U = rand(output_size, output_size)
# d = 0.8

# function rand_img(s)
#     x = rand(0:width-s)
#     y = rand(0:height-s)
#     i = rand(1:dataset_size)
#     return images[1+x:x+s,1+y:y+s,i]'
# end

# while true
    
#     x = reshape(img, :)
#     if all(x .== 0) return end

#     plot(Gray.(rand_img(28)))
#     plot(Gray.(rand_img(patch)))
#     W ./= sum(W, dims=2)

#     s = W * x
#     gamma = SBR(s .> d)
# end

# images = Flux.Data.MNIST.images();
# labels = Flux.Data.MNIST.labels();
# using CUDA;
# CUDA.versioninfo()
