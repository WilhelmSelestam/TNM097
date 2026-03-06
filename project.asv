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
my_256_palette = [ansi; cube; grays];
my_256_palette = my_256_palette(1:256, :); % Final array

%% Visualize the colors 
% 1. Reshape the 256x3 palette into a 16x16 grid
% Since each color is an RGB triplet, the final size is 16x16x3
palette_grid = reshape(my_256_palette, [16, 16, 3]);

% 2. Display the grid
figure;
imshow(palette_grid, 'InitialMagnification', 'fit');
axis on;
title('256 Color Dataset (16x16 Grid)');

% Optional: Label the axes to help identify specific color indices
xlabel('Column');
ylabel('Row');

%% Load Image and get its main color clusters
clc
img_org = im2double(imread("peppers_color.tif"));
%img_org = im2double(imread("sea_sky.jpg"));

% Get dimenssions of input image
[rows, cols, channels] = size(img_org);

% get number of figures in col/row
num_fig_hight = 720/15;
num_fig_width = floor((720 * cols/rows)/15);

img_work = imresize(img_org, [num_fig_hight*15,num_fig_width*15], "nearest");

% Convert original image to CIELAB
img_lab = rgb2lab(img_work);

% Flatten to Nx3 list of pixels
pixel_data_lab = reshape(img_lab, [], 3);

% Shapes

% 15x15
romb_15 = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
           0 0 0 0 0 0 1 1 1 0 0 0 0 0 0;
           0 0 0 0 0 1 1 1 1 1 0 0 0 0 0;
           0 0 0 0 1 1 1 1 1 1 1 0 0 0 0;
           0 0 0 1 1 1 1 1 1 1 1 1 0 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 0 1 1 1 1 1 1 1 1 1 0 0 0;
           0 0 0 0 1 1 1 1 1 1 1 0 0 0 0;
           0 0 0 0 0 1 1 1 1 1 0 0 0 0 0;
           0 0 0 0 0 0 1 1 1 0 0 0 0 0 0;
           0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];

bar_15 = zeros(15, 15);
bar_15(4:12, :) = 1;

% Draw the Mosaic
% Convert mask to logical for easy indexing
mask = logical(bar_15);

% Calculate the ratio of colored pixels vs total pixels in the shape
mask_ratio = sum(mask(:)) / numel(mask); % For romb_12, this is 72/132 (~0.545)

% Create a simulated palette that accounts for the black background
darken_palette_rgb = my_256_palette * mask_ratio;

% Convert this effective palette to LAB for accurate visual matching
darken_palette_3d = reshape(darken_palette_rgb, [256, 1, 3]);
darken_palette_lab = reshape(rgb2lab(darken_palette_3d), [256, 3]);

% The amount of colors in the final palette
num_colors = 64; 

% Perform K-means clustering
[~, image_centroids_lab] = kmeans(pixel_data_lab, num_colors, 'MaxIter', 100, 'EmptyAction', 'drop');

% Map centroids to dataset
idx = knnsearch(darken_palette_lab, image_centroids_lab);

% Remove duplicates 
unique_idx = unique(idx);

% Pull the final colors from the ORIGINAL RGB palette for viewing
repro_palette = my_256_palette(unique_idx, :);

% Check how many unique colors we actually ended up with
actual_num_colors = size(repro_palette, 1);

% Visualize
figure;
subplot(1,2,1); 
imshow(img_org); % Fixed variable name
title('Original Image');

subplot(1,2,2); 
% Create a swatch using the actual number of extracted colors
% repelem creates nice clean blocks without interpolation blur
swatch_blocks = repelem(repro_palette, 20, 20); 
swatch_img = reshape(swatch_blocks, actual_num_colors * 20, 20, 3);

imshow(swatch_img); 
title(sprintf('Optimized Subset (%d Colors)', actual_num_colors));

%% create new image
clc
img_final = zeros(num_fig_hight * 15, num_fig_width * 15, 3);

% We have our actual bright colors (repro_palette). 
% Now we simulate how those specific 36 colors look on a black background.
effective_repro_rgb = repro_palette * mask_ratio;

% Convert this subset to LAB for accurate distance matching inside the loop
effective_repro_3d = reshape(effective_repro_rgb, [actual_num_colors, 1, 3]);
effective_repro_lab = reshape(rgb2lab(effective_repro_3d), [actual_num_colors, 3]);

% create new images 
for i = 1:num_fig_hight
    for j = 1:num_fig_width
        % start and end of block
        row_start = 1 + (i-1)*15;
        row_end   = row_start + 14;
        col_start = 1 + (j-1)*15;
        col_end   = col_start + 14;
        
        % Extract the actual 9x7 RGB block from the resized image
        current_block = img_work(row_start:row_end, col_start:col_end, :);

        % Calculate average color using ONLY the pixels inside the rhombus
        % By using the logical mask, we ignore the background corners
        block_r = current_block(:,:,1);
        block_g = current_block(:,:,2);
        block_b = current_block(:,:,3);
        
        avg_r = mean(block_r(:));
        avg_g = mean(block_g(:));
        avg_b = mean(block_b(:));

        current_block_rgb = zeros(1,1,3);
        current_block_rgb(1,1,1) = avg_r;
        current_block_rgb(1,1,2) = avg_g;
        current_block_rgb(1,1,3) = avg_b;

        current_block_lab = rgb2lab(current_block_rgb);
        avg_lab = squeeze(current_block_lab)';

        % Find the closest color in our optimized LAB subset
        distances = sqrt(sum((effective_repro_lab - avg_lab).^2, 2));
        
        [~, closest_idx] = min(distances);
        best_rgb = repro_palette(closest_idx, :);

        % Paint the image with rombs
        for c = 1:3
        canvas = img_final(row_start:row_end, col_start:col_end, c);
        canvas(mask) = best_rgb(c);
        img_final(row_start:row_end, col_start:col_end, c) = canvas;
        end
    end
end

%% Visualize Final Results
figure('Name', 'Mosaic Results', 'Position', [100, 100, 1200, 500]);

subplot(1,2,1); 
imshow(img_org); 
title('Original Image');

subplot(1,2,2); 
imshow(img_final);
title(sprintf('Rhombus Mosaic (%d Colors)', actual_num_colors));
