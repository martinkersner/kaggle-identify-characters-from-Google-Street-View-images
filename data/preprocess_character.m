% Martin Kersner, m.kersner@gmail.com
% 2015/11/10

% TODO rewrite to python

function R4 = preprocess_character(img_char)
    I = my_rgb2gray(img_char);

    I = padarray(I,[5 5],'replicate','both');

    [rows, cols] = size(I);
    
    R1 = otsu(I);
    R1 = inverse(R1);
    
    R2 = edge(I, 'Canny');
    R2 = dilate(R2);
    R2 = inverse(R2);
    R2 = imfill(R2, 'holes');
    R2 = bwareaopen(R2, total_pixels(R2, 0.1), 4);

    R3 = R1 .* R2;
    R3 = dilate(R3);
    R3 = bwareaopen(R3, total_pixels(R2, 0.1), 4);

    R4 = pad_img(R3);
    R4 = imresize(R4, [256 256], 'nearest');

function R = pad_img(I)
    [rows, cols] = size(I);

    if (rows < cols)
        pad = round((cols-rows) / 2);
        R = [zeros(pad, cols); I; zeros(pad, cols)];
    else
        pad = round((rows-cols) / 2);
        R = [zeros(rows, pad) I zeros(rows, pad)];
    end

function total = total_pixels(I, min_area)
    area = sum(I(:));
    total = double(uint32(area*min_area));

function R = inverse(I)
    offset = 5;
    offset2 = 2*offset;
    
    s = I(1,1) + I(1, end) + I(end, 1) + I(end, end);
    s_offset = I(offset, offset) + I(offset, end-offset) + I(end-offset, offset) + I(end-offset, end-offset);
    s_offset2 = I(offset2, offset2) + I(offset2, end-offset2) + I(end-offset2, offset2) + I(end-offset2, end-offset-2);

    s = s + s_offset + s_offset2;

    if (s > 6)
        R = 1 - I;
    else
        R = I;
    end

function R = dilate(I)
    [rows, cols] = size(I);
    min_size = 50;
    min_size2 = 100;

    if (rows < min_size || cols < min_size)
        strel_size = 1;
    elseif (rows < min_size2 || cols < min_size2)
        strel_size = 3;
    else
        strel_size = 5;
    end

    se = strel('disk', strel_size);
    R = imdilate(I, se);

function R = otsu(I)
    level = graythresh(I);
    R = im2bw(I,level);

function gray = my_rgb2gray(I)
    if (length(size(I)) == 2)
        gray = I;
    else
        gray = rgb2gray(I);
    end
