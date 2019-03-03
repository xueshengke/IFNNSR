function [matrix] = dhtmtx(N)
%DHTMTX Summary of this function goes here
%   Detailed explanation goes here
[k, n] = meshgrid(0 : N-1);
matrix = sqrt(1/N) * ( cos(2*pi/N*n.*k) + sin(2*pi/N*n.*k) );
end