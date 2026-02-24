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

circle = [0 0 0 1 1 0 0 0;
          0 0 1 1 1 1 0 0;
          0 1 1 1 1 1 1 0;
          1 1 1 1 1 1 1 1;
          1 1 1 1 1 1 1 1;
          0 1 1 1 1 1 1 0;
          0 0 1 1 1 1 0 0;
          0 0 0 1 1 0 0 0;];

circle_rgb = zeros(8,8,3);
circle_rgb(:,:,1) = circle;
circle_rgb(:,:,2) = circle;
circle_rgb(:,:,3) = circle;

g = [0 0 0 0 1 0 0 0 0;
     0 0 0 1 1 1 0 0 0;
     0 0 1 1 1 1 1 0 0;
     0 1 1 1 1 1 1 1 0;
     0 1 1 1 1 1 1 1 0;

     0 0 1 1 1 1 1 0 0;
     0 0 0 1 1 1 0 0 0;
     0 0 0 0 1 0 0 0 0;];




[rows, cols, channels] = size(im);
pixel_data = rgb2lab(reshape(im, rows * cols, channels));
num_colors = 100;
[idx, C] = kmeans(pixel_data, num_colors, 'MaxIter', 200);

circle_lab = rgb2lab(circle_rgb);
im_lab = rgb2lab(im);

for i = 1:8:512
    for j = 1:8:512
        optimal_r = sum(sum(im(i:i+7, j:j+7, 1)))/40;
        optimal_g = sum(sum(im(i:i+7, j:j+7, 2)))/40;
        optimal_b = sum(sum(im(i:i+7, j:j+7, 3)))/40;

        current_block_rgb = zeros(1,1,3);
        current_block_rgb(:,:,1) = optimal_r;
        current_block_rgb(:,:,2) = optimal_g;
        current_block_rgb(:,:,3) = optimal_b;

        current_block_lab = rgb2lab(current_block_rgb);

%         current_pixel_lab = reshape(rgb2lab(test), 1, 3);
        current_block_lab = [current_block_lab(:,:,1), current_block_lab(:,:,2), current_block_lab(:,:,3)];

        distances = sqrt(sum((C - current_block_lab).^2, 2));
        
        [~, closest_idx] = min(distances);
        best_lab = C(closest_idx, :);
        best_rgb = lab2rgb(best_lab);

        im2(i:i+7, j:j+7, 1) = circle .* best_rgb(1);
        im2(i:i+7, j:j+7, 2) = circle .* best_rgb(2);
        im2(i:i+7, j:j+7, 3) = circle .* best_rgb(3);
%         im2(i:i+7, j:j+7, 1) = optimal_r;
%         im2(i:i+7, j:j+7, 2) = optimal_g;
%         im2(i:i+7, j:j+7, 3) = optimal_b;
    end
end


% block_size = 8;
% scale_factor = 1 / block_size;
% 
% small_img = imresize(im, scale_factor, 'box');
% [s_rows, s_cols, ~] = size(small_img);
% 
% small_pixels = reshape(small_img, s_rows * s_cols, channels);
% 
% distances = pdist2(small_pixels, C); 
% 
% [~, block_idx] = min(distances, [], 2); 
% 
% quantized_small_pixels = C(block_idx, :);
% quantized_small_img = reshape(quantized_small_pixels, s_rows, s_cols, channels);
% 
% blocky_img = lab2rgb(imresize(quantized_small_img, block_size, 'nearest'));
% 


subplot(1, 2, 1);
imshow(im);

subplot(1, 2, 2);
imshow(im2);




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

circle = [0 0 0 1 1 0 0 0;
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
circle = circle .* target_r_sum/sum(sum(circle))

sum(sum(circle))

%%

clc
clear

im = im2double(imread("peppers_color.tif"));






















