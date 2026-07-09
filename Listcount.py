from collections import Counter
import matplotlib.pyplot as plt

def count_items(lst, savepath=None):
    # Use Counter to count occurrences of each item in the list
    item_counts = Counter(lst)

    items=[]
    counts=[]
    S=""
    for item, count in item_counts.items():
        #print(f"{item}: {count}")
        S+=f"{item}: {count} ,"
        items.append(item)
        counts.append(count)
    #plt.plot(items,counts)
    #print("size of items=", len(items),"size of counts = ",len(counts))
    #plt.plot(torch.arange(len(items)), counts, color='green')
    #plt.plot(items, counts, color='green')
    plt.figure()
    plt.title("Count of Times a State Selected as Start State")
    plt.xlabel("Start States")
    plt.ylabel("Number of Times Selected")
    plt.bar(items, counts, alpha=0.6, color='blue', width=5)
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()
    plt.close()
    #print("S = ",S)
    return items,S
#####################################################################3
def count_items2(lst,number):
    # Use Counter to count occurrences of each item in the list
    item_counts = Counter(lst)
    items_with_count_greater_than_1=[item for item, count in item_counts.items() if count>number]
    items=[]
    counts=[]
    S=""
    for item, count in item_counts.items():
        #print(f"{item}: {count}")
        S+=f"{item}: {count} ,"
        items.append(item)
        counts.append(count)
    return items_with_count_greater_than_1,items,counts

def uniqueIndexes(l):
    seen = set()
    res = []
    for i, n in enumerate(l):
        if n not in seen:
            res.append(i)
            seen.add(n)
    return res
