function [ output ] = idct2Trans( input )
%DCTTRANS Summary of this function goes here
%   Detailed explanation goes here
    output = zeros(size(input));
    if size(input, 3) > 1
        for i = 1 : size(input, 3)
            output(:, : ,i) = idct2(input(:, :, i));
        end
    else
        output = idct2(input);
    end
end
