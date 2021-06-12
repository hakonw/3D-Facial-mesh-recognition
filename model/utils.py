# You can argue that this is stupid
# This function is used to overwrite the pytorch geometric standard
#   collate function  which uses (as of 13.06.2021) a zipping function
#   which restricts the batch to all contain the same amount of scans
#   as the smalest identity.  
def list_collate_fn(batch):
    return batch
