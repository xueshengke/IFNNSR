function batch = imtobatch(image, patch_size)
c = 20;  % cropsize
na = patch_size(1);  % row
nb = patch_size(2);  % column

a = size(image, 1);
b = size(image, 2);

m = ceil(b/(nb - 2*c));
n = ceil(a/(na - 2*c));

p_im1 = padarray(image, [c,c], 'symmetric', 'pre');
p_im = padarray(p_im1, [n*(na-2*c)-a+c, m*(nb-2*c)-b+c], 'symmetric', 'post' );

batch = double(zeros(na, nb, m*n));
for j = 1 : n
    for i = 1 : m
        batch(:, :, i+m*(j-1)) = p_im( (j-1)*(na-2*c)+1 : j*(na-2*c)+2*c , ...
                                       (i-1)*(nb-2*c)+1 : i*(nb-2*c)+2*c);
    end
end

end