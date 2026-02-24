%%% Project
clc
clear
% Generate color palette
% 1. 16 Standard ANSI colors
ansi = [0,0,0; 128,0,0; 0,128,0; 128,128,0; 0,0,128; 128,0,128; 0,128,128; 192,192,192;
        128,128,128; 255,0,0; 0,255,0; 255,255,0; 0,0,255; 255,0,255; 0,255,255; 255,255,255] / 255;

% 2. 216 colors (the 6x6x6 color cube)
vals = [0, 95, 135, 175, 215, 255] / 255;
[R, G, B] = ndgrid(vals, vals, vals);
cube = [R(:), G(:), B(:)];

% 3. 24 Grayscale levels (gradient from dark to light)
grays_val = linspace(8, 238, 24)' / 255;
grays = [grays_val, grays_val, grays_val];

% Combine into the final 256x3 dataset
my_palette = [ansi; cube; grays];
my_palette = my_palette(1:256, :); % Final array

%% Visualize the colors 
% 1. Reshape the 256x3 palette into a 16x16 grid
% Since each color is an RGB triplet, the final size is 16x16x3
palette_grid = reshape(my_palette, [16, 16, 3]);

% 2. Display the grid
figure;
imshow(palette_grid, 'InitialMagnification', 'fit');
axis on;
title('256 Color Dataset (16x16 Grid)');

% Optional: Label the axes to help identify specific color indices
xlabel('Column');
ylabel('Row');

%%
im = im2double(imread("peppers_color.tif"));
im2 = im;

% 8x8
circle = [0 0 0 1 1 0 0 0;
          0 0 1 1 1 1 0 0;
          0 1 1 1 1 1 1 0;
          1 1 1 1 1 1 1 1;
          1 1 1 1 1 1 1 1;
          0 1 1 1 1 1 1 0;
          0 0 1 1 1 1 0 0;
          0 0 0 1 1 0 0 0;];

% 9x9
romb = [0 0 0 0 1 0 0 0 0;
        0 0 0 1 1 1 0 0 0;
        0 0 1 1 1 1 1 0 0;
        0 1 1 1 1 1 1 1 0;
        1 1 1 1 1 1 1 1 1;
        0 1 1 1 1 1 1 1 0;
        0 0 1 1 1 1 1 0 0;
        0 0 0 1 1 1 0 0 0;
        0 0 0 0 1 0 0 0 0;];


[rows, cols, channels] = size(im);
pixel_data = reshape(im, rows * cols, channels);
num_colors = 10;
[idx, C] = kmeans(pixel_data, num_colors, 'MaxIter', 150);

block_size = 8;
scale_factor = 1 / block_size;

small_img = imresize(im, scale_factor, 'box');
[s_rows, s_cols, ~] = size(small_img);

small_pixels = reshape(small_img, s_rows * s_cols, channels);

distances = pdist2(small_pixels, C); 

[~, block_idx] = min(distances, [], 2); 

quantized_small_pixels = C(block_idx, :);
quantized_small_img = reshape(quantized_small_pixels, s_rows, s_cols, channels);

blocky_img = imresize(quantized_small_img, block_size, 'nearest');

subplot(1, 2, 1);
imshow(im);

subplot(1, 2, 2);
imshow(blocky_img);

%%

for i = 1:8:512
    for j = 1:8:512
        r2 = r .* sum(sum(im(i:i+7, j:j+7, 1)))/sum(sum(r));
        g2 = g .* sum(sum(im(i:i+7, j:j+7, 2)))/sum(sum(g));
        b2 = b .* sum(sum(im(i:i+7, j:j+7, 3)))/sum(sum(b));

        im2(i:i+7, j:j+7, 1) = r2;
        im2(i:i+7, j:j+7, 2) = g2;
        im2(i:i+7, j:j+7, 3) = b2;
    end
end

%%

 subplot(1,2,1)
 imshow(im)
 
 subplot(1,2,2)
 imshow(im2)

 quantized_pixel_data = C(idx, :);
 quantized_img = reshape(quantized_pixel_data, rows, cols, channels);

 subplot(1, 2, 1);
 imshow(im); 
 
 subplot(1, 2, 2);
 imshow(quantized_img);

%%
clc
clear

r = [0 0 0 1 1 0 0 0;
          0 0 1 1 1 1 0 0;
          0 1 1 1 1 1 1 0;
          1 1 1 1 1 1 1 1;
          1 1 1 1 1 1 1 1;
          0 1 1 1 1 1 1 0;
          0 0 1 1 1 1 0 0;
          0 0 0 1 1 0 0 0;];

im = im2double(imread("peppers_color.tif"));

im(1:8, 1:8, 1);

target_r_sum = sum(sum(im(1:8, 1:8, 1)))
r = r .* target_r_sum/sum(sum(r))

sum(sum(r))

%%

clc
clear

im = im2double(imread("peppers_color.tif"));






















