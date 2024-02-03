
def _get_image_list(path):
  ext_list = ['png', 'jpg', 'jpeg']
  image_list = []
  for ext in ext_list:
    image_list.extend(glob.glob(f'{path}/*{ext}'))
    image_list.extend(glob.glob(f'{path}/*.{ext.upper()}'))
  # Sort the list to ensure a deterministic output.
  image_list.sort()
  return image_list


def _center_crop_and_resize(im, size):
  w, h = im.size
  l = min(w, h)
  top = (h - l) // 2
  left = (w - l) // 2
  box = (left, top, left + l, top + l)
  im = im.crop(box)
  # Note that the following performs anti-aliasing as well.
  return im.resize((size, size), resample=Image.BICUBIC)  # pytype: disable=module-attr


def _read_image(path, reshape_to):
  im = Image.open(path)
  if reshape_to > 0:
    im = _center_crop_and_resize(im, reshape_to)
  return np.asarray(im).astype(np.float32)

class ImageDataset(Dataset):
    def __init__(self, img_paths, reshape_to):
        self.img_paths = img_paths
        self.reshape_to = reshape_to

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = _read_image(img_path, self.reshape_to)
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        return transforms.ToTensor()(image)

def compute_embeddings_for_dir(img_dir, embedding_model, batch_size, max_count=-1):
    img_path_list = _get_image_list(img_dir)
    if max_count > 0:
        img_path_list = img_path_list[:max_count]

    dataset = ImageDataset(img_path_list, embedding_model.input_image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embs = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            embs = embedding_model(batch).cpu().numpy()
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs