function output = hartleyTrans(input, method)
%% Hartley transform does not comput through Fourier transform, which will 
%% be wrong
dims = ndims(input);
if dims == 1
    N = size(input);
    output = dhtmtx(N) * input;
end

if dims >= 2
    M = size(input, 1);
    N = size(input, 2);
    C = size(input, 3);
    for i = 1 : C
        if isequal(method, 't')
            output(:,:,i) = dhtmtx(M) * input(:,:,i) * dhtmtx(N);
            output(:,:,i) = fftshift(output(:,:,i));
        elseif isequal(method, 'i')
            input(:,:,i) = ifftshift(input(:,:,i));
            output(:,:,i) = dhtmtx(M) * input(:,:,i) * dhtmtx(N);
        end
        
    end
end
end