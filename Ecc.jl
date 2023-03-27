using LinearAlgebra
using Random
using Test
using StatsBase
using ProgressBars

function dense(indices::Union{AbstractVector{<:Integer},Integer}, shape)
    binary_vec = Array{Bool}(undef, shape)
    return dense!(indices, binary_vec)
end
function dense!(indices::Union{AbstractVector{<:Integer},Integer}, dense::AbstractArray{Bool})
    fill!(dense, false)
    for index = indices
        dense[index] = true
    end
    return dense
end
function sparse(binary_vector::AbstractArray{Bool})
    indices = Vector{Int32}()
    for (index, bool_val) = enumerate(binary_vector)
        if bool_val
            push!(indices, index)
        end
    end
    return indices
end

"""
image is of shape [channels, height, width]
"""
function rand_patch(image::AbstractArray, patch_height::Integer, patch_width::Integer)
    e = ndims(image)
    h = size(image, e-1)
    w = size(image, e)
    patch_height -= 1
    patch_width -= 1  # stupid Julia uses 1-based indexing. 
    y = rand(1:h-patch_height)
    x = rand(1:w-patch_width)
    if e == 2
        return @view image[y:y+patch_height, x:x+patch_width]
    else
        return @view image[:, y:y+patch_height, x:x+patch_width]
    end
end
"""
dataset is of shape [channels, height, width, batch]
"""
function rand_img(dataset::AbstractArray)
    e = ndims(dataset)
    return selectdim(dataset,e,rand(1:size(dataset, e)))
end
"""
generate random sparse binary vector of length n with exactly k non-zero entries
"""
function rand_sparse_k(k::Int, n)
    n = prod(n)
    if k > n 
        error("k=$(k) > $(n)=n")
    end
    return sample(1:n , k, replace=false)
end
"""
generate random binary vector of length n with exactly k non-zero entries
"""
function rand_dense_k(k, n)
    return dense(rand_sparse_k(k, n), n)
end
"""
returns indices i at which dense[i] > threshold. Those indices are sorted by dense[i] in descending order.
"""
function sparse_gt(dense::AbstractArray, threshold)
    dense = vec(dense)
    binary = threshold .< dense
    sparse = findall(binary)
    sparse = sparse[sortperm(dense[sparse], rev=true)]
    return sparse
end
"""
returns k indices corresponding to the k highest values in dense vector. Those indices i are sorted by dense[i] in descending order.
"""
function sparse_top_k(dense::AbstractArray, k::Integer)
    return partialsortperm(dense, 1:k, rev=true)
end
"""
returns binary vector with exactly k ones at places corresponding to the k highest values in dense_x vector.
"""
function dense_top_k(dense_x::AbstractArray, k::Integer)
    return dense(sparse_top_k(dense_x, k), size(dense_x))
end
function dense_top_k!(dense_x::AbstractArray, k::Integer, dense_y::AbstractArray)
    @assert size(dense_x) == size(dense_y)
    return dense!(sparse_top_k(dense_x, k), dense_y)
end
function sparse_inner_product(sparse_vec::AbstractVector{<:Integer}, dense::AbstractVector)
    return sum(dense[sparse_vec])
end
function sparse_dot(sparse_vec::AbstractVector{<:Integer}, dense::AbstractMatrix)
    o = similar(dense, size(dense, 2))
    return sparse_dot!(sparse_vec, dense, o)
end
function sparse_dot!(sparse_vec::AbstractVector{<:Integer}, dense::AbstractMatrix, output::AbstractVector)
    for j in axes(dense, 2)
        output[j] = sparse_inner_product(sparse_vec, dense[:, j])
    end
    return output
end
function sparse_dot_t(dense::AbstractMatrix, sparse_vec::AbstractVector{<:Integer})
    o = similar(dense, size(dense, 1))
    return sparse_dot_t!(dense, sparse_vec, o)
end
function sparse_dot_t!(dense::AbstractMatrix, sparse_vec::AbstractVector{<:Integer}, output::AbstractVector)
    for j in axes(dense, 1)
        output[j] = sparse_inner_product(sparse_vec, dense[j, :])
    end
    return output
end
function ordered_swta_v(v::AbstractMatrix{Bool}, s::AbstractVector, si::AbstractVector{<:Integer})
    y = fill(2, size(s))
    return ordered_swta_v!(v,s,si,y)
end
function ordered_swta_v!(v::AbstractMatrix{Bool}, s::AbstractVector, si::AbstractVector{<:Integer}, y::AbstractVector{<:Int})
    @assert size(y) == size(s)
    for k ∈ si
        if y[k] == 2
            y[k] = 1
            for j ∈ eachindex(s)
                if y[j] == 2 && v[k, j] && s[k] > s[j]
                    y[j] = 0
                end
            end
        end
    end
    return y
end
function swta_v(v::AbstractMatrix{Bool}, s::AbstractVector)
    return ordered_swta_v(v, s, sortperm(s, rev=true))
end
function swta_v!(v::AbstractMatrix{Bool}, s::AbstractVector, y::AbstractVector{<:Int})
    return ordered_swta_v!(v, s, sortperm(s, rev=true), y)
end
function ordered_swta_u(u::AbstractMatrix, s::AbstractVector, si::AbstractVector{<:Integer})
    y = fill(2, size(s))
    return ordered_swta_u!(u,s,si,y)
end
function ordered_swta_u!(u::AbstractMatrix, s::AbstractVector, si::AbstractVector{<:Integer}, y::AbstractVector{<:Int})
    for k ∈ si
        if y[k] == 2
            y[k] = 1
            for j ∈ eachindex(s)
                if y[j] == 2 && s[k] > s[j] + u[k, j]
                    y[j] = 0
                end
            end
        end
    end
    return y
end

function soft_wta_u(u::AbstractMatrix, s::AbstractVector)
    return ordered_swta_u(u, s, sortperm(s, rev=true))
end
function soft_wta_u!(u::AbstractMatrix, s::AbstractVector, y::AbstractVector{<:Int})
    return ordered_swta_u!(u, s, sortperm(s, rev=true), y)
end


module conv
    using LinearAlgebra
    """
    The beginning of input range (inclusive). This is the first input neuron that connects to the specified output neuron. USES 0-BASED INDEXING 
    Example with stride=1 kernel=2
    out = [y0, y1, y2,  ...  y(m-1)]
         / \\ /\\ /\\          \\
    in=[x0, x1, x2, x3, x4, ... x(n-1)]
    In genral the pattern is as follows:
    in_range_begin(0) = 0
    in_range_begin(1) = stride
    in_range_begin(2) = 2*stride
    """
    function in_range_begin(out_position::Tuple, stride::Tuple) 
        out_position .* stride
    end

    """The end of input range (exclusive). This is the last input neuron that connects to the specified output neuron. USES 0-BASED INDEXING"""
    function in_range_end(out_position::Tuple, stride::Tuple, kernel_size::Tuple)
        return in_range_begin(out_position, stride).+kernel_size
    end

    """returns the range of inputs that connect to a specific output neuron. USES 0-BASED INDEXING"""
    function in_range(out_position::Tuple, stride::Tuple, kernel_size::Tuple)
        from = in_range_begin(out_position, stride)
        to = from.+kernel_size
        return from, to
    end

    """
    returns the range of inputs that connect to any neuron within some patch of output neuron.
    That output patch starts (inclusive) at position specified by out_position and ends at out_position+output_patch_size (inclusive)
    """
    function in_range_with_custom_size(out_position::Tuple, output_patch_size::Tuple, stride::Tuple, kernel_size::Tuple)
        if all(output_patch_size.>0) 
            from = in_range_begin(out_position, stride)
            to = in_range_end(out_position+output_patch_size-1, stride, kernel_size)
            return from, to
        else 
            return zero(out_position), zero(out_position)
        end
    end

    """returns the range of outputs that connect to a specific input neuron"""
    function out_range(in_position::Tuple, stride::Tuple, kernel_size::Tuple)
        """
        Notation:
        A .. B represents a range {i : A <= i < B}
        A ..= B represents a range {i : A <= i <= B}

        Derivation:
        out_position * stride .. out_position * stride + kernel
        out_position * stride ..= out_position * stride + kernel - 1
        
        in_position_from == out_position * stride
        in_position_from / stride == out_position
        round_down(in_position / stride) == out_position_to
        
        in_position_to == out_position * stride + kernel - 1
        (in_position_to +1 - kernel)/stride == out_position
        round_up((in_position +1 - kernel)/stride) == out_position_from
        round_down((in_position +1 - kernel + stride - 1)/stride) == out_position_from
        round_down((in_position - kernel + stride)/stride) == out_position_from
        
        (in_position - kernel + stride)/stride ..= in_position / stride
        (in_position - kernel + stride)/stride .. in_position / stride + 1
        """
        to = in_position.÷stride.+1;
        from = (in_position.+stride.-kernel_size).÷stride;
        return from, to
    end
    """
    Calculates the transpose kernel size. Kernel size normally tells you how many input neurons are connected to a single output neuron.
    Transpose kernel says how many output neurons are connected to the same input neuron. If kernel > stride then the transpose kernel is undefined.
    """
    function out_transpose_kernel(kernel::Tuple, stride::Tuple)
        """
        (in_position - kernel + stride)/stride .. in_position / stride + 1
        in_position / stride + 1 - (in_position - kernel + stride)/stride
        (in_position- (in_position - kernel + stride))/stride + 1
        (kernel - stride)/stride + 1
        """
        @assert all(kernel .< stride)
        return (kernel.-stride).÷stride .+ 1
    end

    """
    returns the range of outputs that connect to a specific input neuron.
    output range is clipped to 0, so that you don't get overflow on negative values when dealing with unsigned integers.
    """
    function out_range_clipped(in_position::Tuple, stride::Tuple, kernel_size::Tuple)
        to = (in_position .÷ stride).+1
        from = (max.(in_position.+stride, kernel_size).-kernel_size).÷stride
        return from, to
    end

    function out_range_clipped_both_sides(in_position::Tuple, stride::Tuple, kernel_size::Tuple, max_bounds::Tuple)
        from, to = out_range_clipped(in_position, stride, kernel_size)
        return from, min.(to, max_bounds)
    end

    function out_size(input::Tuple, stride::Tuple, kernel_size::Tuple)
        @assert all(kernel_size .<= input) "Kernel size $(kernel_size) is larger than the input shape $(input)";
        input_sub_kernel = input.-kernel_size
        @assert all(input_sub_kernel .% stride .== 0 ) "Convolution stride $(stride) does not evenly divide the input shape $(input)-$(kernel_size)=$(input_sub_kernel)"
        return input_sub_kernel .÷ stride.+1
        # (input-kernel)/stride+1 == output
    end

    function in_size(output::Tuple, stride::Tuple, kernel_size::Tuple)
        @assert all(output .> 0), "Output size $(output) contains zero"
        return (output.-1).*stride.+kernel_size
        #input == stride*(output-1)+kernel
    end

    function stride(input::Tuple, out_size::Tuple, kernel_size::Tuple)
        @assert all(kernel_size .<= input) "Kernel size $(kernel_size) is larger than the input shape $(input)"
        input_sub_kernel = input.-kernel_size
        out_size_minus_1 = out_size.-1
        @assert all(input_sub_kernel .% out_size_minus_1 .== 0) "Output shape $(out_size)-1 does not evenly divide the input shape $(input)"
        return input_sub_kernel.÷out_size_minus_1
        #(input-kernel)/(output-1) == stride
    end

    function compose(self_stride::Tuple, self_kernel::Tuple, next_stride::Tuple, next_kernel::Tuple)
        """
        (A-kernelA)/strideA+1 == B
        (B-kernelB)/strideB+1 == C
        ((A-kernelA)/strideA+1-kernelB)/strideB+1 == C
        (A-kernelA+(1-kernelB)*strideA)/(strideA*strideB)+1 == C
        (A-(kernelA-(1-kernelB)*strideA))/(strideA*strideB)+1 == C
        (A-(kernelA+(kernelB-1)*strideA))/(strideA*strideB)+1 == C
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^                    composed kernel
                                        ^^^^^^^^^^^^^^^ composed stride
        """
        composed_kernel = (next_kernel-.1).*self_stride .+ self_kernel
        composed_stride = self_stride.*next_stride
        return composed_stride, composed_kernel
    end

    function range_contains(range::Tuple{Tuple,Tuple}, pos::Tuple)
        start_inclusive, end_exclusive = range
        return all(start_inclusive .<= pos .< end_exclusive)
    end 

    struct Shape
        #[in_channels, in_height, in_width]
        input_shape::Tuple{Integer,Integer,Integer}
        #[out_channels, out_height, out_width]
        output_shape::Tuple{Integer,Integer,Integer}
        #[kernel_height, kernel_width]
        kernel::Tuple{Integer,Integer}
        #[height, width]
        stride::Tuple{Integer,Integer}
        Shape(input_shape,output_shape,kernel,stride) = grid(output_shape) == out_size(grid(input_shape),stride,kernel) ? new(input_shape,output_shape,kernel,stride) : error("Input $(input_shape) and output $(output_shape) shapes are incompatible")
    end
    """
    Produces a new convolutional Shape from a given input shape
    """
    function shape(input_shape::Tuple{Integer,Integer,Integer}, out_channels::Int, kernel::Tuple{Integer,Integer}, stride::Tuple{Integer,Integer} = (1,1))
        out_y, out_x = out_size(grid(input_shape),stride,kernel)
        return Shape(input_shape, (out_channels, out_y, out_x), kernel, stride)
    end
    """
    Produces a new convolutional Shape from a given output shape
    """
    function shape_reverse(in_channels::Int, output_shape::Tuple{Integer,Integer,Integer}, kernel::Tuple{Integer,Integer}, stride::Tuple{Integer,Integer} = (1,1))
        in_y, in_x = in_size(grid(input_shape),stride,kernel)
        return Shape((in_channels, in_y, in_x), output_shape, kernel, stride)
    end
    function in_dense(indices::AbstractVector{<:Integer}, shape::Shape)
        return dense(indices, shape.input_shape)
    end
    function out_dense(indices::AbstractVector{<:Integer}, shape::Shape)
        return dense(indices, shape.output_shape)
    end
    """[out_channels, out_height, out_width, in_channels, kernel_height, kernel_width]"""
    function w_shape(shape::Shape)
        kernel_height, kernel_width = shape.kernel
        out_channels, out_height, out_width = shape.output_shape
        return (out_channels, out_height, out_width, in_channels(shape), kernel_height, kernel_width)
    end
    """[out_channels, in_channels, kernel_height, kernel_width]"""
    function w_conv_shape(shape::Shape)
        kernel_height, kernel_width = shape.kernel
        return (out_channels(shape), in_channels(shape), kernel_height, kernel_width)
    end
    """[out_channels, out_channels, out_height, out_width]"""
    function u_shape(shape::Shape)
        out_height, out_width, out_channels = shape.output_shape
        return (out_channels, out_channels, out_height, out_width)
    end
    """[out_channels, out_channels]"""
    function u_conv_shape(shape::Shape)
        c = out_channels(shape)
        return (c, c)
    end
    function grid(pos::Tuple)
        return length(pos) >= 3 ? pos[2:end] : pos 
    end
    function in_range(shape::Shape, output_pos::Tuple)
        return in_range(grid(output_pos), shape.stride, shape.kernel)
    end
    function out_range(shape::Shape, input_pos::Tuple)
        return out_range_clipped_both_sides(grid(input_pos), shape.stride, shape.kernel, grid(shape.output_shape))
    end
    function kernel_offset(shape::Shape, output_pos::Tuple) 
        return in_range_begin(grid(output_pos), shape.stride)
    end
    function sub_kernel_offset(input_pos::Tuple{Integer,Integer,Integer}, offset::Tuple{Integer,Integer}) 
        ic, iy, ix = input_pos
        oy, ox = offset
        return (ic, iy-oy, ix-ox)
    end
    function pos_within_kernel!(shape::Shape, input_pos::Tuple{Integer,Integer,Integer}, output_pos::Tuple{Integer,Integer,Integer}) 
        @assert all(0 .<= output_pos .< self.output_shape)
        @assert all(0 .<= input_pos .< self.input_shape)
        @assert range_contains(in_range(shape, output_pos), grid(input_pos))
        @assert range_contains(out_range(shape, input_pos), grid(output_pos))
        return sub_kernel_offset!(input_pos, kernel_offset(shape, output_pos))
    end
    function pos_within_kernel(shape::Shape, input_pos::Tuple{Integer,Integer,Integer}, output_pos::Tuple{Integer,Integer,Integer}) 
        return pos_within_kernel!(shape, copy(input_pos), output_pos)
    end
    """[in_height, in_width]"""
    function in_grid(shape::Shape)
        return grid(shape.input_shape)
    end
    """[out_height, out_width]"""
    function out_grid(shape::Shape)
        return grid(shape.output_shape)
    end
    function out_channels(shape::Shape)
        return shape.output_shape[1]
    end
    function out_height(shape::Shape)
        return shape.output_shape[2]
    end
    function out_width(shape::Shape)
        return shape.output_shape[3]
    end
    function in_channels(shape::Shape)
        return shape.input_shape[1]
    end
    function in_height(shape::Shape)
        return shape.input_shape[2]
    end
    function in_width(shape::Shape)
        return shape.input_shape[3]
    end
    """out_height * out_width"""
    function out_area(shape::Shape)
        return prod(out_grid(shape))
    end
    """in_height * in_width"""
    function in_area(shape::Shape)
        return prod(in_grid(shape))
    end
    """out_height * out_width * out_channels"""
    function out_volume(shape::Shape)
        return prod(shape.output_shape)
    end
    """in_height * in_width * in_channels"""
    function in_volume(shape::Shape)
        return prod(shape.input_shape)
    end 
    """[in_channels, kernel_height, kernel_width]"""
    function kernel_column_shape(shape::Shape)
        kh, kw = shape.kernel
        return (in_channels(shape), kh, kw)
    end
    """*in_channels*kernel_height*kernel_width"""
    function kernel_column_volume(shape::Shape)
        return prod(kernel_column_shape(shape))
    end
    """
    dense_w is of shape [out_channels, out_height, out_width, in_channels, kernel_height, kernel_width]
    kernel_col is of shape [out_channels, in_channels, kernel_height, kernel_width]
    Repeats the same weights [out_channels, in_channels, kernel_height, kernel_width] across all kernel column [out_height, out_width]
    """
    function repeat_kernel!(shape::Shape, dense_w::AbstractArray, kernel_col::AbstractArray)
        @assert size(kernel_col) == w_conv_shape(shape) "$(size(kernel_col)) != $(w_conv_shape(shape))"
        @assert size(dense_w) == w_shape(shape) "$(size(dense_w)) != $(w_shape(shape))"
        for out_x in axes(dense_w, 2)
            for out_y in axes(dense_w, 3)
                dense_w[:,out_y,out_x,:,:,:] .= kernel_col
            end
        end
        return dense_w
    end
    function repeat_kernel(shape::Shape, kernel_col::AbstractArray)
        w = similar(kernel_col, w_shape(shape))
        return repeat_kernel!(shape, w, kernel_col)
    end
    function sparse_dot(shape::Shape, sparse_x::AbstractVector{<:Integer}, dense_w::AbstractArray) 
        out = zeros(shape.output_shape)
        return sparse_dot!(shape, sparse_x, dense_w, out)
    end
    """
    Performs convolutional dot product between a sparse matrix (left) and dense convolutional filter (right).
    dense_w is of shape [out_channels, out_height, out_width, in_channels, kernel_height, kernel_width]
    """
    function sparse_dot!(shape::Shape, sparse_x::AbstractVector{<:Integer}, dense_w::AbstractArray, out::AbstractArray) 
        @assert size(dense_w) == w_shape(shape) "$(size(dense_w)) != $(w_shape(shape))"
        @assert size(out) == shape.output_shape "$(size(out)) != $(shape.output_shape)"
        cart_indices = CartesianIndices(Tuple(0:i-1 for i in shape.input_shape))
        for input_idx ∈ sparse_x
            input_pos = Tuple(cart_indices[input_idx])
            from_out_pos, to_out_pos = out_range(shape, input_pos)
            from_out_y, from_out_x = from_out_pos 
            to_out_y, to_out_x = to_out_pos
            for x ∈ from_out_x:to_out_x-1
                for y ∈ from_out_y:to_out_y-1
                    kernel_offset = in_range_begin((y, x), shape.stride)
                    in_c, position_within_kernel_column_y, position_within_kernel_column_x = sub_kernel_offset(input_pos, kernel_offset)
                    # println("y=$(y), x=$(x), in_c=$(in_c), position_within_kernel_column=($(position_within_kernel_column_y), $(position_within_kernel_column_x)) input_pos=$(input_pos) kernel_offset=$(kernel_offset)")
                    out[:, y+1, x+1] .+= dense_w[:, y+1, x+1, in_c+1, position_within_kernel_column_y+1, position_within_kernel_column_x+1]
                end
            end
        end
        return out
    end
    function sparse_conv(shape::Shape, sparse_x::AbstractVector{<:Integer}, dense_w::AbstractArray) 
        out = zeros(shape.output_shape)
        return sparse_conv!(shape, sparse_x, dense_w, out)
    end
    """
    Performs convolutional dot product between a sparse matrix (left) and dense convolutional filter (right).
    dense_w is of shape [out_channels, in_channels, kernel_height, kernel_width]
    """
    function sparse_conv!(shape::Shape, sparse_x::AbstractVector{<:Integer}, dense_w::AbstractArray, out::AbstractArray) 
        @assert size(dense_w) == w_conv_shape(shape) "$(size(dense_w)) != $(w_conv_shape(shape))"
        @assert size(out) == shape.output_shape "$(size(out)) != $(shape.output_shape)"
        cart_indices = CartesianIndices(Tuple([0:i-1 for i in shape.input_shape]))
        for input_idx ∈ sparse_x
            input_pos = Tuple(cart_indices[input_idx])
            from_out_pos, to_out_pos = out_range(shape, grid(input_pos))
            from_out_y, from_out_x = from_out_pos 
            to_out_y, to_out_x = to_out_pos
            for x ∈ from_out_x:to_out_x-1
                for y ∈ from_out_y:to_out_y-1
                    kernel_offset = in_range_begin((y,x), shape.stride)
                    in_c, position_within_kernel_column_y, position_within_kernel_column_x = sub_kernel_offset(input_pos, kernel_offset)
                    out[:, y+1, x+1] .+= dense_w[:, in_c+1, position_within_kernel_column_y+1, position_within_kernel_column_x+1]
                end
            end
        end
        return out
    end
    function dot(shape::Shape, dense_x::AbstractArray, dense_w::AbstractArray)
        out = similar(dense_w, shape.output_shape)
        return dot!(shape, dense_x, dense_w, out)
    end
    """
    dense_w is of shape [out_channels, out_height, out_width, in_channels, kernel_height, kernel_width]
    dense_x is of shape [in_channels, in_height, in_width]
    """
    function dot!(shape::Shape, dense_x::AbstractArray, dense_w::AbstractArray, out::AbstractArray)
        @assert size(dense_x) == shape.input_shape
        @assert size(dense_w) == w_shape(shape)
        @assert size(out) == shape.output_shape "$(size(out)) != $(shape.output_shape)"
        for out_x ∈ axes(out,3)
            for out_y ∈ axes(out,2)
                for out_c ∈ axes(out,1)
                    out_val = 0
                    kernel_oy, kernel_ox = kernel_offset(shape, (out_y, out_x).-1)
                    for kernel_y ∈ axes(dense_w, 5)
                        for kernel_x ∈ axes(dense_w, 6)
                            in_y = kernel_oy+kernel_y
                            in_x = kernel_ox+kernel_x
                            out_val += dense_x[:,in_y,in_x] ⋅ dense_w[out_c,out_y,out_x,:,kernel_y,kernel_x]
                        end
                    end
                    out[out_c,out_y,out_x] = out_val
                end
            end
        end
        return out
    end

end


module layer
    import ..rand_img
    import ..rand_patch
    import ..sparse
    import ..sparse_gt
    import ..dense
    import ..sparse_top_k
    import ..sparse_dot
    using LinearAlgebra
    

    abstract type Layer end

    function n(layer::Layer)
        return layer.n
    end

    function m(layer::Layer)
        return layer.m
    end

    mutable struct HardWtaL2 <: Layer
        const r_step::Real
        W::AbstractMatrix{Real}
        r::AbstractVector{Real}
        const norm
        const w_step::Real
        const n::Int
        const m::Int
        const min_input_cardinality::Int
    end  

    function hard_wta_l2(n::Int,m::Int,l=2, w_step=0.0001, r_step=1. / 1024 * 2, min_input_cardinality::Int=1)
        W = rand(n, m)
        r = zeros(m)
        if l==2
            norm_func = norm
        elseif l==1
            norm_func = sum
        else
            error("L$(l) norm is not implemented")
        end
        W = mapslices(x -> x / norm_func(x), W, dims=1)
        return HardWtaL2(r_step,W,r,norm_func,w_step,n,m,min_input_cardinality)
    end

    function run(ecc::HardWtaL2, x::AbstractVector{<:Integer})
        y = Vector{Int}()
        if length(x) >= ecc.min_input_cardinality
            push!(y, argmax(sparse_dot(x, ecc.W) + ecc.r))
        end
        return y
    end

    function learn(ecc::HardWtaL2, x::AbstractVector{<:Integer}, k::AbstractVector{<:Integer})
        ecc.r[k] .-= ecc.r_step
        ecc.W[x, k] .+= ecc.w_step / sum(x)
        ecc.W[:, k] ./= ecc.norm(@view ecc.W[:,k])
    end

    mutable struct HardWtaL1 <: Layer
        const r_step::Real
        Q::AbstractMatrix{Real}
        W_sparse::AbstractMatrix{Integer}
        W_dense::AbstractMatrix{Bool}
        r::AbstractVector{Real}
        const q_step::Real
        const n::Integer
        const m::Integer
        const a::Integer
    end  

    function hard_wta_l1(n::Integer,m::Integer,a::Integer,q_step=0.0001, r_step=1. / 1024 * 2)
        @assert a <= n "a=$(a) > $(n)=n"
        Q = rand(n, m)
        r = zeros(m)
        W_sparse = mapslices(x -> sparse_top_k(x, a), Q, dims=1)
        W_dense = mapslices(x -> dense(x, n), W_sparse, dims=1)
        @assert (a, m) == size(W_sparse)
        @assert size(Q) == size(W_dense)
        return HardWtaL1(r_step,Q,W_sparse,W_dense,r,q_step,n,m,a)
    end

    function run(ecc::HardWtaL1, x::AbstractVector{<:Integer})
        y = Vector{Int}()
        if length(x) >= ecc.min_input_cardinality
            push!(y, argmax(sparse_dot(x, ecc.W_dense) + ecc.r))
        end
        return y
    end

    function learn(ecc::HardWtaL1, x::AbstractVector{<:Integer}, k::Integer)
        ecc.r[k] -= ecc.r_step
        ecc.Q[:, k] .*= (1. .- ecc.q_step)
        ecc.Q[x, k] .+= ecc.q_step
        ecc.W_dense[ecc.W_sparse[:, k], k] .= false
        ecc.W_sparse[:, k] .= sparse_top_k(ecc.Q[:, k], ecc.a)
        ecc.W_dense[ecc.W_sparse[:, k], k] .= true
        @assert sum(ecc.W_dense[:,k]) == ecc.a
    end
end


"""
dataset is of shape [channels, height, width, batch]
"""
function train_on_patches(ecc_layer::layer.Layer, dataset::AbstractArray{Bool}, num_images::Integer, patches_per_image::Integer, patch_height, patch_width)
    channels = ndims(dataset) == 4 ? size(dataset, 1) : 1
    @assert channels * patch_height * patch_width == ecc_layer.n "c×h×w=$(channels)×$(patch_height)×$(patch_width)=$(channels * patch_height * patch_width) != $(ecc_layer.n)=n"
    pb = tqdm(1:num_images)
    set_description(pb, "train_on_patches")
    for _ ∈ pb
        img = rand_img(dataset)
        for _ ∈ 1:patches_per_image
            x = rand_patch(img, patch_height, patch_width)
            x = sparse(x)
            y = layer.run(ecc_layer, x)
            layer.learn(ecc_layer, x, y)
        end
    end
end
function stack_dense(shape, iterable)
    dense = Array{Bool}(undef, shape)
    for (sparse, dense_slice) in zip(iterable, eachslice(dense, dims=length(shape)))
        dense!(sparse, dense_slice)
    end
    return dense
end
"""
inputs is a list of sparse vectors. Returns a list of sparse outputs.
"""
function run_sparse(ecc_layer::layer.Layer, inputs::Vector{Vector{Integer}})
    pb = tqdm(inputs)
    set_description(pb, "run_sparse")
    return [layer.run(ecc_layer, x) for x in pb]
end
"""
inputs is a list of sparse vectors. Returns a binary array [m, BATCH]
"""
function run_dense(ecc_layer::layer.Layer, inputs::Vector{Vector{Integer}})
    pb = tqdm(inputs)
    set_description(pb, "run_dense")
    return stack_dense((ecc_layer.m, length(inputs)), layer.run(ecc_layer, x) for x in pb)
end

function run_dense_conv(ecc_layer::layer.Layer, shape::conv.Shape, inputs::AbstractArray{Bool})
    out = Array{Bool}(undef, shape.output_shape)
    return run_dense_conv!(ecc_layer, shape, inputs, out)
end
"""
x is of shape [in_channels, in_height, in_width].
Applies an ECC network across an entire image. It works exactly like convolution with weight-sharing. Thus every ECC network can be used as a convolutional ECC network.
The power of hebbian learning is that you can train a single ECC network on small image patches and that is equivalent to training an entire convolutional ECC network on whole images.
Unlike deep nets, we don't need backprop, hence, we don't need to evalue whole network to compute any gradients. We can train individual subnetworks (convolutional filters) independently.
So just train the ecc_layer first and once its trained you can use run_dense_conv for inference on whole images. 
"""
function run_dense_conv!(ecc_layer::layer.Layer, shape::conv.Shape, inputs::AbstractArray, out::AbstractArray{Bool, 3})
    @assert layer.n(ecc_layer) == conv.kernel_column_volume(shape) "$(layer.n(ecc_layer)) != $(conv.kernel_column_volume(shape))"
    @assert layer.m(ecc_layer) == conv.out_channels(shape) "$(layer.m(ecc_layer)) != $(conv.out_channels(shape))"
    @assert size(inputs) == shape.input_shape "$(size(inputs)) != $(shape.input_shape)"
    @assert size(out) == shape.output_shape "$(size(out)) != $(shape.output_shape)"
    kernel_y, kernel_x = shape.kernel.-1  # stupid julia uses 1-based indexing
    stride_y, stride_x = shape.stride
    fill!(out, false)
    out_x = 1
    for x ∈ 1:stride_x:conv.in_width(shape)-kernel_x
        out_y = 1
        for y ∈ 1:stride_y:conv.in_height(shape)-kernel_y
            sparse_kernel_column = sparse(inputs[:, y:y+kernel_y, x:x+kernel_x])
            sparse_out = layer.run(ecc_layer, sparse_kernel_column)
            for i in sparse_out
                out[i, out_y, out_x] = true
            end
            out_y += 1
        end
        out_x += 1
    end
    return out
end

function run_sparse_conv(ecc_layer::layer.Layer, shape::conv.Shape, inputs::AbstractArray)
    @assert layer.n(ecc_layer) == conv.kernel_column_volume(shape) "$(layer.n(ecc_layer)) != conv.kernel_column_volume(shape)"
    @assert layer.m(ecc_layer) == conv.out_channels(shape) "$(layer.m(ecc_layer)) != conv.out_channels(shape)"
    @assert size(inputs) == shape.input_shape "$(size(inputs)) != $(shape.input_shape)"
    kernel_y, kernel_x = shape.kernel.-1  # stupid julia uses 1-based indexing
    stride_y, stride_x = shape.stride
    cart_to_lin = LinearIndices(shape.output_shape) 
    out = Vector{Int}()
    out_x = 1
    for x ∈ 1:stride_x:conv.in_width(shape)-kernel_x
        out_y = 1
        for y ∈ 1:stride_y:conv.in_height(shape)-kernel_y
            sparse_kernel_column = sparse(inputs[:, y:y+kernel_y, x:x+kernel_x])
            offset::Int = cart_to_lin[1, out_y, out_x]
            sparse_out = offset - 1 .+ layer.run(ecc_layer, sparse_kernel_column)
            append!(out, sparse_out)
            out_y += 1
        end
        out_x += 1
    end
    return out
end
"""
x is of shape [in_channels, in_height, in_width, batch]
"""
function batch_run_dense_conv(ecc_layer::layer.Layer, shape::conv.Shape, inputs::AbstractArray{Bool})
    @assert ndims(inputs) == 4
    outputs = Array{Bool}(undef, shape.output_shape..., size(inputs, 4))
    pb = tqdm(zip(eachslice(inputs, dims=4), eachslice(outputs, dims=4)))
    set_description(pb, "batch_run_dense_conv")
    for (x, y) in pb
        run_dense_conv!(ecc_layer, shape, x, y) 
    end
    return outputs 
end
"""
x is of shape [in_channels, in_height, in_width, batch]
"""
function batch_run_sparse_conv(ecc_layer::layer.Layer, shape::conv.Shape, inputs::AbstractArray{Bool})
    @assert ndims(inputs) == 4
    pb = tqdm(eachslice(inputs, dims=4))
    set_description(pb, "batch_run_sparse_conv")
    return [run_sparse_conv(ecc_layer, shape, x) for x in pb]
end

module head
    using ProgressBars
    mutable struct NaiveBayes
        const in_size::Int
        const num_classes::Int
        # [num_classes, in_size]
        # number of times each input bit was activated in presence of each label
        counts::Matrix{Int}  
        # [num_classes]
        # number of times each label occured
        class_counts::Vector{Int}
        # How many samples were presented to the naive bayes in total 
        total_num_samples::Int
        p_of_class_a_priori::Vector{Float32}
        log_p_words_and_class
        NaiveBayes(in_size::Int, num_classes::Int) = new(in_size, num_classes, zeros(num_classes, in_size), zeros(num_classes), 0, )
    end
    function naive_bayes(in_size::Int, num_classes::Int, sparse_datatset::AbstractVector{<:AbstractVector{<:Integer}}, labels::AbstractVector{<:Integer})
        return naive_bayes!(NaiveBayes(in_size, num_classes), sparse_datatset, labels)
    end
    function naive_bayes!(nb::NaiveBayes, sparse_datatset::AbstractVector{<:AbstractVector{<:Integer}}, labels::AbstractVector{<:Integer})
        pb = tqdm(zip(sparse_datatset, labels))
        set_description(pb, "training naive_bayes")
        for (x, y) in pb
            for i in x
                nb.counts[y, i] += 1
            end
            nb.class_counts[y] += 1
        end
        nb.total_num_samples += length(labels)
        return nb
    end
    function run(nb::NaiveBayes, sparse::AbstractVector{<:Integer})
        log_p = log.(nb.class_counts) .- log.(nb.total_num_samples) # = log p(y)
        for i in sparse
            # p(x_i=1 | y) = {number of occurences of x[i] together with label y} / {number of occurences of y}
            log_p .+= log.(nb.counts[:, i].+1) .- log.(nb.class_counts.+nb.num_classes) # = log p(x_i1, x_i2, ..., x_ik, y)
        end
        # p(x,y) = p(x_i1 | y) p(x_i2 | y) ... p(x_ik | y) p(y)
        return argmax(log_p)
    end
    function batch_run(nb, sparse)
        pb = tqdm(sparse)
        set_description(pb, "head.batch_run")
        return [run(nb, y) for y in pb] 
    end
end
####################################################################################
###############################      TESTS      ####################################
####################################################################################

function test1()
    cs = conv.shape((4,32,32),4,(5,5),(1,1))
    x = rand_sparse_k(256, cs.input_shape)
    xx = dense(x, Tuple(cs.input_shape))
    w = rand(conv.w_shape(cs)...)
    o = conv.sparse_dot(cs,x,w)
    oo = conv.dot(cs,xx,w)
    @assert all(o.-oo.<0.000001) "$(o.-oo)"
    return true
end

function test2()
    cs = conv.shape((4,32,32),4,(5,5),(1,1))
    x = rand_sparse_k(256, cs.input_shape)
    w_conv = rand(conv.w_conv_shape(cs)...)
    w = conv.repeat_kernel(cs, w_conv)
    o = conv.sparse_dot(cs,x,w)
    o_conv = conv.sparse_conv(cs,x,w_conv)
    @assert all(o.-o_conv.<0.000001) "$(o.-oo)"
    return true
end

function test3()
    oc = 4
    ph, pw = 5, 5
    ic = 4
    ecc = layer.hard_wta_l2(ic*ph*pw, oc)
    cs = conv.shape((ic,32,32),oc,(5,5),(1,1))
    x = rand_dense_k(256, cs.input_shape)
    y_dense = run_dense_conv(ecc, cs, x)
    y_sparse = run_sparse_conv(ecc, cs, x)
    y_dense2 = dense(y_sparse, cs.output_shape)
    @assert all(y_dense2.==y_dense) "$(y_dense2.==y_dense)"
    return true
end

function test4()
    oc = 4
    ph, pw = 4, 4
    ic = 4
    ecc = layer.hard_wta_l2(ic*ph*pw, oc)
    cs = conv.shape((ic,32,32),oc,(ph,pw),(2,2))
    x = rand_dense_k(256, cs.input_shape)
    y_dense = run_dense_conv(ecc, cs, x)
    y_sparse = run_sparse_conv(ecc, cs, x)
    @assert length(y_sparse)>0 "$(length(y_sparse))"
    y_dense2 = dense(y_sparse, cs.output_shape)
    @assert all(y_dense2.==y_dense) "$(y_dense2.==y_dense)"
    return true
end

function test5()
    x = [[1],[2],[3],[4],[5]]
    y = [1,2,3,4,5]
    nb = head.naive_bayes(5, 5, x, y)
    y2 = head.batch_run(nb, x)
    @assert all(y2.==y) "$(y2) != $(y)"
    return true
end

function test6()
    X::Vector{Vector{Int}} = [[1,2,3], [3,4,5], [1,4,6], []]
    ecc = layer.hard_wta_l2(6, 3)
    for _ in 1:30
        for x in X
            y = layer.run(ecc, x)
            layer.learn(ecc, x, y)
        end
    end
    y1 = layer.run(ecc,X[1])
    y2 = layer.run(ecc,X[2])
    y3 = layer.run(ecc,X[3])
    @assert length(layer.run(ecc, X[4])) == 0
    @assert length(y1) == 1 "$(y1) != $(y2) != $(y3)"
    @assert length(y2) == 1 "$(y1) != $(y2) != $(y3)"
    @assert length(y3) == 1 "$(y1) != $(y2) != $(y3)"
    @assert y1[1]!=y2[1] "$(y1) != $(y2) != $(y3)"
    @assert y2[1]!=y3[1] "$(y1) != $(y2) != $(y3)"
    @assert y3[1]!=y1[1] "$(y1) != $(y2) != $(y3)"
    @assert 1 <= y1[1]  <= ecc.m "$(y1) != $(y2) != $(y3)"
    @assert 1 <= y2[1]  <= ecc.m "$(y1) != $(y2) != $(y3)"
    @assert 1 <= y3[1]  <= ecc.m "$(y1) != $(y2) != $(y3)"
    return true
end

@testset "Conv Tests" begin
    @test test1()
    @test test2()
    @test test3()
    @test test4()
    @test test5()
    @test test6()
end