function output = quarterSplit( input, border )
    if ~exist('border', 'var'), border = 0; end
    [hei, wid, num] = size(input);
    
    quadrant_1 = input(1:hei/2+border, wid/2+1-border:end, :);
    quadrant_1 = reshape(quadrant_1, [hei/2+border  wid/2+border  1  num]);
    
    quadrant_2 = input(1:hei/2+border, 1:wid/2+border, :);
    quadrant_2 = fliplr(quadrant_2); %
    quadrant_2 = reshape(quadrant_2, [hei/2+border  wid/2+border  1  num]);
    
    quadrant_3 = input(hei/2+1-border:end, 1:wid/2+border, :);
    quadrant_3 = fliplr(flipud(quadrant_3)); %
    quadrant_3 = reshape(quadrant_3, [hei/2+border  wid/2+border  1  num]);
    
    quadrant_4 = input(hei/2+1-border:end, wid/2+1-border:end, :);
    quadrant_4 = flipud(quadrant_4); %
    quadrant_4 = reshape(quadrant_4, [hei/2+border  wid/2+border  1  num]);
    
    output = cat(3, quadrant_1, quadrant_2, quadrant_3, quadrant_4);
end