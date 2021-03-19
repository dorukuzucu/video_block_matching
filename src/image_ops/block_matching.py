import numpy as np
from itertools import product

from src.metrics import image_mse


class Block:
    """
    Block object with center point and data which is pixels in are represented
    """
    def __init__(self, center, data, block_size):
        self.center = center
        self.data = data
        self.block_size = block_size


class BlockList:
    def __init__(self,inp_list=None):
        if inp_list is not None and len(inp_list)>1 and  isinstance(inp_list, list):
            self.list = inp_list
        else:
            self.list = []

    def add(self,block):
        if isinstance(block,Block):
            self.list.append(block)
        else:
            print("Please input a block type variable")

    def get_min_score(self,score="mse"):
        min_score = float('Inf')
        block_with_min_score = None

        for block in self.list:
            if block.data<min_score:
                block_with_min_score = block
                min_score = block.data

        return block_with_min_score

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]


class ThreeStepSearch:
    """
    This class implements 3 step search algorithm:
        -for a given anchor block
        -in a given target image
    """
    def __init__(self, step_size):
        self.step_size = step_size

    def _generate_block_centers(self, search_center, step_size):
        """
        :param search_center: center main block to be searched in target
        :return: All center candidates
        """
        step_options = [0, -1*step_size, step_size]
        steps = list(product(step_options,step_options))

        centers = []
        for step_element in steps:
            new_center = np.add(search_center,step_element)
            centers.append(new_center)

        centers = np.array(centers)
        centers[centers<0]=0

        return centers

    def __call__(self, input_block: Block, target):
        """
        :param input_block: block to be searched from frame at time t=t1-1
        :param target: target frame at time t=t1
        :return: best fit Block object in target frame
        """
        assert input_block.data.shape[0]==input_block.data.shape[1], "Input block must be a square!"

        block_size = input_block.data.shape[0]
        step_size = self.step_size

        while step_size>1:
            block_centers = self._generate_block_centers(search_center=input_block.center,step_size=step_size)
            block_list = BlockList()

            for block_center in block_centers:
                if (block_center[0]-block_size//2)<0 or (block_center[1]-block_size//2)<0:
                    continue
                if ((block_center[0]+block_size//2)>=target.shape[0]) or ((block_center[1]+block_size//2)>=target.shape[1]):
                    continue
                try:
                    block_data = target[(block_center[0]-block_size//2):(block_center[0]+block_size//2),
                            (block_center[1]-block_size//2):(block_center[1]+block_size//2)]

                except Exception:
                    print("Block range exceeds target image dimensions")
                    continue

                score = image_mse(input_block.data, block_data)
                block_list.add(Block(center=block_center, data=score,block_size=block_size))

            input_block = block_list.get_min_score()

            min_center = input_block.center
            min_data = target[(min_center[0] - block_size // 2):(min_center[0] + block_size // 2),
                           (min_center[1] - block_size // 2):(min_center[1] + block_size // 2)]

            input_block = Block(center=min_center,data=min_data,block_size=block_size)
            step_size = step_size//2

        return input_block


class BlockMatchFrames:
    def __init__(self, block_size, step_size, metric=None):
        self.block_size = block_size
        self.metric = metric
        self.step_size = step_size

        self.three_step_searcher = ThreeStepSearch(step_size=self.step_size)

    def _generate_blocks(self,image,height_block_count,width_block_count):
        block_list = BlockList()

        for h in range(height_block_count):
            for w in range(width_block_count):

                block_center_h = h*self.block_size+self.block_size//2
                block_center_w = w*self.block_size+self.block_size//2

                block_data =  image[(block_center_h-self.block_size//2):(block_center_h+self.block_size//2),
                            (block_center_w-self.block_size//2):(block_center_w+self.block_size//2)]

                block_list.add(Block(center=[block_center_h,block_center_w],data=block_data,block_size=self.block_size))

        return block_list

    def __call__(self, anchor, target):
        if type(anchor) is not np.ndarray:
            raise Exception("Please input a numpy nd array")
        if type(target) is not np.ndarray:
            raise Exception("Please input a numpy nd array")

        assert anchor.shape == target.shape, "Frame shapes do not match"
        assert len(anchor.shape)==2, "Frames should be gray"

        height, width = anchor.shape

        block_count_h = height//self.block_size
        block_count_w = width//self.block_size

        anchor_block_list = self._generate_blocks(anchor,block_count_h,block_count_w)

        block_pairs_to_swap = []

        for anchor_block in anchor_block_list:
            matched_block = self.three_step_searcher(input_block=anchor_block,target=target)
            block_pairs_to_swap.append((anchor_block,matched_block))

        return block_pairs_to_swap

