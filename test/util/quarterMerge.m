function output = quarterMerge( input, border )
if ~exist('border', 'var'), border = 0; end
[hei, wid, ch, num] = size(input);
hei = hei - border;
wid = wid - border;
output = zeros(hei*sqrt(ch), wid*sqrt(ch), num);
%     output(1:hei,     wid+1:end, :) = input(:, :, 1, :);
%     output(1:hei,     1:wid,     :) = input(:, :, 2, :);
%     output(hei+1:end, 1:wid,     :) = input(:, :, 3, :);
%     output(hei+1:end, wid+1:end, :) = input(:, :, 4, :);

%     output(1:hei,     wid+1:end, :) = input(:, :, 1, :);
%     output(1:hei,     1:wid,     :) = fliplr(input(:, :, 2, :));
%     output(hei+1:end, 1:wid,     :) = fliplr(flipud(input(:, :, 3, :)));
%     output(hei+1:end, wid+1:end, :) = flipud(input(:, :, 4, :));

output(1:hei,     wid+1:end, :) =               input(1:end-border, 1+border:end, 1, :);
output(1:hei,     1:wid,     :) = fliplr(       input(1:end-border, 1+border:end, 2, :));
output(hei+1:end, 1:wid,     :) = fliplr(flipud(input(1:end-border, 1+border:end, 3, :)));
output(hei+1:end, wid+1:end, :) = flipud(       input(1:end-border, 1+border:end, 4, :));
end