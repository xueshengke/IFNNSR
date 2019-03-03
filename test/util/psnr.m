function [psnr, MSE] = psnr(image_1, image_2, omega_c, max_val)
%PSNR Summary of this function goes here
%   Detailed explanation goes here

if ~exist('max_val', 'var')
    max_val = 255;
end

T = length(find(omega_c == 1));

diff = (double(image_1) - double(image_2)) .* omega_c;

erec = norm(diff(:))^2;

MSE = erec / T;

psnr = 10 * log10(max_val^2 / MSE);

end

