#include "iostream"
#include "cstring"
#include "string"
#include "tbb/parallel_for.h"
#include "tbb/concurrent_queue.h"
#include "tbb/task_scheduler_init.h"
#include "map"
#include "algorithm"
#include "set"
#include "vector"
#include "queue"
#include "sstream"
#include "cstdint"
#include "sys/time.h"
#include "fcntl.h"
#include "nmmintrin.h"//SSE4.2(include smmintrin.h) 

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

using namespace std;
using namespace tbb;
const uint64_t GRAM_TOTAL_LEN = 4000000000LL;
const int QUERY_DOC_LEN = 200000100;
const int MAX_LINE_LEN = 1000000;
const int BUFFER_LEN = 1048576;
const int THREADNUM = 40;
const int M = 50100;
const int MIN_COND_LEN = 3;
const int MIN_COND_LEVEL = 2;
char line_buffer[MAX_LINE_LEN];

char *stdin_buf, *stdout_buf;
char *read_buffer;
char *input_word;
int input_id = 0;

struct TrieNode;
struct TrieTree;

struct QueryOp
{
    int begin;    
    int end;
    int timestamp;

    QueryOp() {}
    QueryOp(int _b, int _e, int _ts): begin(_b), end(_e), timestamp(_ts) {}
};
QueryOp query_list[M];
int query_cnt = 0;

const int TRIE_TREE_NUM = 60;

int hash_tree_idx_map[256][256];
int init_gram_head_freq[256][256], total_hash_count[TRIE_TREE_NUM];
vector<char*> init_grams;

struct HeadFreq
{
    int h1, h2, freq;
    HeadFreq(int _h1, int _h2, int _f): h1(_h1), h2(_h2), freq(_f) {}
    bool operator < (const HeadFreq& p) const {
        return freq > p.freq;
    }
};

void gen_hash_tree_idx_map()
{
    vector<HeadFreq> hf_vec;
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j) {
            hf_vec.push_back(HeadFreq(i, j, init_gram_head_freq[i][j]));
        }
    sort(hf_vec.begin(), hf_vec.end());
    int rotate_idx = 0;
    for (int i = 0; i < hf_vec.size(); ++ i) {
        if (hf_vec[i].freq == 0) {
            hash_tree_idx_map[hf_vec[i].h1][hf_vec[i].h2] = rotate_idx;
            // rotate_idx = (rotate_idx + 1) % TRIE_TREE_NUM;
            // if (rotate_idx == 0) rotate_idx++;
        } else {
            int idx = rotate_idx;
            for (int j = 0; j < TRIE_TREE_NUM; ++ j) {
                if (total_hash_count[j] < total_hash_count[idx]) {
                    idx = j;
                }
            }
            hash_tree_idx_map[hf_vec[i].h1][hf_vec[i].h2] = idx;
            total_hash_count[idx] += hf_vec[i].freq;           
        }
        rotate_idx = (rotate_idx + 1) % TRIE_TREE_NUM;
    }
}

int calc_tree_idx(const char* s)
{
    unsigned char h1 = (unsigned char)s[0];
    unsigned char h2 = s[1] == '\0' ? (unsigned char)' ' : (unsigned char)s[1];
    return hash_tree_idx_map[h1][h2];
}

int insert_block_query_idx[TRIE_TREE_NUM][M];
int insert_block_cnt[TRIE_TREE_NUM];

struct TrieChild
{
    char value;
    TrieNode* ptr;
    TrieChild() {}
    TrieChild(char _val, TrieNode* _c_ptr): value(_val), ptr(_c_ptr) {}
};

struct TrieNode
{
    TrieChild* child; // child array.
    char *str; // point to the head of inserted gram.
    char *cond_str; // point to the rest of the gram if it is a condensed node.
    int cond_len;
    int global_query_idx;
    uint16_t size, capa; // child num, child capacity.
    uint16_t pos; // index of update vector in a batch.
    char node_state; // 0 for non-leaf node, 1 for end_state, 2 for cond_state(condensed path node), 3 for both end&cond.

    TrieNode* find(char idx)
    {
        for (int i = 0; i < size; ++i)
            if (child[i].value == idx) return child[i].ptr;
        return NULL;
    }

    TrieChild* find2(char idx)
    {
        for (int i = 0; i < size; ++i)
            if (child[i].value == idx) return &child[i];
        return NULL;
    }

    bool is_child_vec_full() {return (size == capa);}

    void add(TrieNode* son_ptr, char son_idx)
    {
        child[size++] = TrieChild(son_idx, son_ptr);
    }

    bool update(TrieNode* son_ptr, char son_idx)
    {
        for (int i = 0; i < size; ++i)
            if (child[i].value == son_idx) {
                child[i].ptr = son_ptr;
                return true;
            }      
        return false;
    }
};

const int TRIE_NODE_POOL_LEN = 40000000;
const int TRIE_CHILD_POOL_LEN = 80000000;
const int INIT_CHILD_CAPACITY = 2;
struct TrieTree
{
    TrieNode* node_pool = NULL;
    int node_cnt = 0;
    TrieChild* child_pool = NULL;
    int child_cnt = 0;
    TrieNode* root = NULL;

    void init()
    {
        node_pool = new TrieNode[TRIE_NODE_POOL_LEN];
        child_pool = new TrieChild[TRIE_CHILD_POOL_LEN];
        node_cnt = child_cnt = 0;
        root = new_node();
    }

    TrieNode* new_node()
    {
        TrieNode* res = node_pool + (node_cnt++);
        res->child = child_pool + child_cnt; child_cnt += INIT_CHILD_CAPACITY;
        res->capa = INIT_CHILD_CAPACITY; res->size = 0;
        res->node_state = 0;
        return res;
    }

    void extend_child_vec(TrieNode* node)
    {
        int new_capa = node->capa * 2;
        TrieChild* new_child_ptr = child_pool + child_cnt;
        child_cnt += new_capa;
        memcpy(new_child_ptr, node->child, sizeof(TrieChild) * (node->size));
        node->child = new_child_ptr;
        node->capa = new_capa;
    }
};

TrieTree trie_tree[TRIE_TREE_NUM];

TrieNode* trie_entry[256][256];

void init_trie_entry()
{
    char head[2];
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
        {
            head[0] = (char)i;
            head[1] = (char)j;
            int tree_idx = calc_tree_idx(head);
            TrieTree& tree = trie_tree[tree_idx];
            TrieNode* entry_node = tree.root;
            if (entry_node->find(head[0]) == NULL) {
                TrieNode* son = tree.new_node();
                if (entry_node->is_child_vec_full()) tree.extend_child_vec(entry_node);
                entry_node->add(son, head[0]);
            }
            entry_node = entry_node->find(head[0]);
            if (entry_node->find(head[1]) == NULL) {
                TrieNode* son = tree.new_node();
                if (entry_node->is_child_vec_full()) tree.extend_child_vec(entry_node);
                entry_node->add(son, head[1]);
            }
            entry_node = entry_node->find(head[1]);
            trie_entry[i][j] = entry_node;
        }
}

struct PrevNodeState
{
    TrieNode* node;
    int node_state;

    PrevNodeState() {}
    PrevNodeState(TrieNode* _ptr, int _ns): node(_ptr), node_state(_ns) {}
};
PrevNodeState prev_state[M];

struct UpdateOp
{
    char *gram_str;
    int gram_len;
    int timestamp;
    int op; // 1 for add, 0 for delete
    TrieNode* node;

    UpdateOp() {}
    UpdateOp(char* _str, int _len, int _ts, int _op):
        gram_str(_str), gram_len(_len), timestamp(_ts), op(_op), node(NULL) {}
};
UpdateOp update_list[M];
int update_cnt;

void Ready()
{
    puts("R");
    cout.flush();
}

// int calc_common_prefix(char *s1, char *s2, int cond_len)
// {
//     for (int i = 0; i < cond_len; ++i)
//         if (s1[i] != s2[i]) return i;
//     return cond_len;
// }
inline int calc_common_prefix(char *s1, char *s2, int cond_len)
{
    const int mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_BIT_MASK | _SIDD_NEGATIVE_POLARITY;
    __m128i smm1 = _mm_loadu_si128 ((__m128i *) s1);
    __m128i smm2 = _mm_loadu_si128 ((__m128i *) s2);
    int length = 0;
    for (;  length < cond_len; length += 16) {
        int len =  _mm_cmpistri (smm1, smm2, static_cast<char>(mode) );
        if (len < 16) return min(length + len, cond_len);
        s1 += 16;
        s2 += 16;
        smm1 = _mm_loadu_si128 ((__m128i *) s1);
        smm2 = _mm_loadu_si128 ((__m128i *) s2);
    }
    return min(length, cond_len);
}

bool Insert(char *gram_str, int gram_len, TrieTree& tree) 
{
    TrieNode *current = tree.root;

    int i = 0;
    while (i < gram_len) {
        char idx = gram_str[i];
        TrieChild* child_node = current->find2(idx);
        TrieNode* son;
        if (child_node == NULL){
            son = tree.new_node();
            if (current->is_child_vec_full()) tree.extend_child_vec(current);
            current->add(son, idx);
            // if (i >= MIN_COND_LEVEL && gram_len - i >= MIN_COND_LEN) {
            if (i >= MIN_COND_LEVEL) {
                son->str = gram_str;
                son->node_state = 1;
                if (gram_len - i > 1) {
                    son->node_state = 3;
                    son->cond_str = gram_str + i + 1;
                    son->cond_len = gram_len - i - 1;
                }
                return true;
            } else {
                current = son;
                i++;
            }
        } else {
            son = child_node->ptr;
            if (son->node_state & 2) {
                int common_prefix_len = calc_common_prefix(gram_str + i + 1, son->cond_str, son->cond_len);
                if (common_prefix_len == son->cond_len) {
                    current = son;
                    i += common_prefix_len + 1;
                } else {
                    TrieNode* new_son = tree.new_node();
                    child_node->ptr = new_son; // update
                    if (common_prefix_len > 0) {
                        new_son->node_state = 2;
                        new_son->cond_str = gram_str + i + 1;
                        new_son->cond_len = common_prefix_len;
                    }
                    char cond_head_ch = *(son->cond_str + common_prefix_len);
                    (son->cond_str) += (common_prefix_len + 1);
                    (son->cond_len) -= (common_prefix_len + 1);
                    new_son->add(son, cond_head_ch);
                    if (son->cond_len == 0) son->node_state &= 1;
                    current = new_son;
                    i += common_prefix_len + 1;
                }
            } else {
                current = son;
                i++;
            }
        }
    }    
    
    if (current->node_state & 1) return false;
    current->node_state |= 1;
    current->str = gram_str;
    return true;
}

TrieNode* InsertInBatch(char* gram_str, int gram_len, int tree_idx) 
{
    TrieTree& tree = trie_tree[tree_idx];
    TrieNode *current = tree.root;
    int i = 0;
    if (gram_len >= 2) {
        current = trie_entry[(unsigned char)gram_str[0]][(unsigned char)gram_str[1]];
        i = 2;
    }

    while (i < gram_len) {
        char idx = gram_str[i];
        TrieChild* child_node = current->find2(idx);
        TrieNode* son;
        if (unlikely(child_node == NULL)) {            
            son = tree.new_node();
            if (current->is_child_vec_full()) tree.extend_child_vec(current);
            current->add(son, idx);
            // if (i >= MIN_COND_LEVEL && gram_len - i >= MIN_COND_LEN) {
            if (likely(i >= MIN_COND_LEVEL)) {                
                son->str = gram_str;
                if (gram_len - i > 1) {
                    son->node_state = 2;
                    son->cond_str = gram_str + i + 1;
                    son->cond_len = gram_len - i - 1;   
                }          
                return son;
            } else {                
                current = son;
                i++;
            }
        } else {            
            son = child_node->ptr;
            if (unlikely(son->node_state & 2)) {                
                int common_prefix_len = calc_common_prefix(gram_str + i + 1, son->cond_str, son->cond_len);
                if (likely(common_prefix_len == son->cond_len)) {                    
                    current = son;
                    i += common_prefix_len + 1;
                } else {                    
                    TrieNode* new_son = tree.new_node();
                    child_node->ptr = new_son; // update
                    if (common_prefix_len > 0) {
                        new_son->node_state = 2;
                        new_son->cond_str = gram_str + i + 1;
                        new_son->cond_len = common_prefix_len;
                    }
                    char cond_head_ch = *(son->cond_str + common_prefix_len);
                    new_son->add(son, cond_head_ch);                    
                    (son->cond_len) -= (common_prefix_len + 1);                    
                    if (son->cond_len == 0) son->node_state &= 1;
                    else (son->cond_str) += (common_prefix_len + 1);

                    i += common_prefix_len + 1;
                    current = new_son;
                    if (i < gram_len) {
                        TrieNode* rest_node = tree.new_node();
                        new_son->add(rest_node, *(gram_str + i));
                        if (gram_len - i > 1) {
                            rest_node->node_state = 2;
                            rest_node->cond_str = gram_str + i + 1;
                            rest_node->cond_len = gram_len - i - 1;
                        }
                        current = rest_node;
                    }
                    current->str = gram_str;
                    return current;
                }
            } else {                
                current = son;
                i++;
            }              
        }
    }    
    
    current->str = gram_str;
    return current;
}

void insert_diff_trie(int tree_idx)
{
    for (int i = 0; i < insert_block_cnt[tree_idx]; ++i)
    {
        int update_list_idx = insert_block_query_idx[tree_idx][i];
        UpdateOp& update_op = update_list[update_list_idx];
        update_op.node = InsertInBatch(update_op.gram_str, update_op.gram_len, tree_idx);
    }
}

class ConcurrentInsertProcessor
{
public:
    void operator() (const blocked_range<size_t> &r) const
    {
        for (size_t i=r.begin();i!=r.end();++i)
        {
            insert_diff_trie(i);
        }
    }
    ConcurrentInsertProcessor(){}
};

int read_buffer_len = 0;
int ReadBatchOperation()
{
    size_t max_line_cnt = QUERY_DOC_LEN;
    bool is_first;
    int timestamp = 0;
    char op;

    memset(insert_block_cnt, 0, sizeof(int) * TRIE_TREE_NUM);
    while((read_buffer_len = getdelim(&read_buffer, &max_line_cnt, 'F', stdin)) != -1) {
        char *current, *current_start = read_buffer;
        if (getchar() == EOF) break;
        while(true) {
            char ch = (*current_start);
            (*current_start) = '\0';
            if (ch == 'F') {
                return 1;
            }
            current = (char*)memchr(current_start, '\n', QUERY_DOC_LEN);
            int length = current - current_start - 2;
            if (ch == 'D' || ch == 'A') {
                char *current_pos = &input_word[input_id];
                memcpy(current_pos, current_start + 2, length + 1);
                input_id += length + 1;
                *(current_pos + length) = '\0';
                int tree_idx = calc_tree_idx(current_pos);
                insert_block_query_idx[tree_idx][insert_block_cnt[tree_idx]++] = update_cnt;
                update_list[update_cnt++] = UpdateOp(current_pos, length, timestamp, ch == 'A');
                timestamp++;
            }else if(ch == 'Q') {
                int query_begin = current_start - read_buffer + 1;
                int query_end = current - read_buffer + 1;
                *current = ' ';
                query_list[query_cnt++] = QueryOp(query_begin, query_end, timestamp);
                timestamp++;       
            } 
            current_start = current + 1;
        }        
    }

    return 0;
}

void reserve(TrieNode* node, TrieTree& tree)
{
    if (node->size == node->capa) tree.extend_child_vec(node);
    for (int i = 0; i < node->size; ++i)
        reserve(node->child[i].ptr, tree);
}

int character_freq[256];
bool charater_freq_cmp(const TrieChild& a, const TrieChild& b)
{
    return character_freq[(unsigned char)a.value] > character_freq[(unsigned char)b.value];
}

void traverse_trie(TrieNode* node, int level)
{
    if (node->size == 0) return;
    sort(node->child, node->child + node->size, charater_freq_cmp);
    for (int i = 0; i < node->size; ++i)
        traverse_trie(node->child[i].ptr, level + 1);
}

void ReadInit()
{
    while(true) {
        gets(line_buffer);
        if (line_buffer[0] == 'S') break;
        else {
            char *current_pos = &input_word[input_id];
            int length = strlen(line_buffer);
            strcpy(current_pos, line_buffer);
            input_id += length + 1;
            init_grams.push_back(current_pos);
            unsigned char head_1 = (unsigned char)current_pos[0];
            unsigned char head_2 = current_pos[1] == '\0' ? (unsigned char)' ' : (unsigned char)current_pos[1];
            init_gram_head_freq[head_1][head_2]++;
        }
    }

    gen_hash_tree_idx_map();
    for (auto current_pos : init_grams) {
        int tree_idx = calc_tree_idx(current_pos);
        int length = strlen(current_pos);
        Insert(current_pos, length, trie_tree[tree_idx]);
    }

    init_trie_entry();

    for (int i = 0; i < TRIE_TREE_NUM; ++i)
        reserve(trie_tree[i].root, trie_tree[i]);

    for (int i = 0; i < input_id; ++i)
        if (input_word[i]) character_freq[(unsigned char)input_word[i]]++;
    for (int i = 0 ; i < TRIE_TREE_NUM; ++i) traverse_trie(trie_tree[i].root, 0);
}

const int BLOCK_SIZE = 256;
const int MAX_BLOCK_NUM = 2048;
struct BlockInterval
{
    int first, last;
    BlockInterval() {}
    BlockInterval(int _f, int _l): first(_f), last(_l) {}
};
BlockInterval block_interval[MAX_BLOCK_NUM];
int block_interval_cnt;

void split_into_block()
{
    int block_size = BLOCK_SIZE;

    while (true) {
        block_interval_cnt = 0;
        bool is_split_success = true;
        for (int i = 0; i < query_cnt && is_split_success; ++i) {
            int cur = query_list[i].begin + 1;
            int this_query_first_block_idx = block_interval_cnt;
            while (cur < query_list[i].end && is_split_success) {
                int next = min(cur + block_size, query_list[i].end);
                if ((next - cur) * 4 < block_size && block_interval_cnt > this_query_first_block_idx) 
                    block_interval[block_interval_cnt - 1].last = next;
                else if (block_interval_cnt < MAX_BLOCK_NUM)
                    block_interval[block_interval_cnt++] = BlockInterval(cur, next);
                else
                    is_split_success = false;                    
                cur = next;
            }
        }
        if (is_split_success) break;
        block_size *= 2;
    }
}

struct Match
{
    int begin;
    int end;
    TrieNode* node_ptr;

    Match() {}
    Match(int _b, int _end, TrieNode* _ptr): begin(_b), end(_end), node_ptr(_ptr) {}
};
const int MAX_MATCH_LEN = 5000000;
const int MATCH_BLOCK_LEN = 50000;
Match* match_block[MAX_BLOCK_NUM];
int match_block_count[MAX_BLOCK_NUM];
Match matches[MAX_MATCH_LEN], valid_matches[MAX_MATCH_LEN];

void AllocMem()
{
    Match* match_block_ptr = new Match[(uint64_t)MAX_BLOCK_NUM * MATCH_BLOCK_LEN];
    for (int i = 0; i < MAX_BLOCK_NUM; ++i)
        match_block[i] = match_block_ptr + ((uint64_t)i * MATCH_BLOCK_LEN);
}

// inline bool isMatchCondNode(char* query_str, char* gram_str, int gram_len)
// {
//     for (int i = 0; i < gram_len; ++i)
//         if (query_str[i] != gram_str[i])
//             return false;
//     return true;
// }
inline bool isMatchCondNode(char* query_str, char* gram_str, int gram_len)
{
    const int mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_BIT_MASK | _SIDD_NEGATIVE_POLARITY;
    __m128i smm1 = _mm_loadu_si128 ((__m128i *) query_str);
    __m128i smm2 = _mm_loadu_si128 ((__m128i *) gram_str);
    // gram_len -= 16;
    while(gram_len > 16) {
        int len =  _mm_cmpistri (smm1, smm2, static_cast<char>(mode) );
        if (len < 16) return false;
        query_str += 16;
        gram_str += 16;
        smm1 = _mm_loadu_si128 ((__m128i *) query_str);
        smm2 = _mm_loadu_si128 ((__m128i *) gram_str);
        gram_len -= 16;
    }
    return _mm_cmpistri (smm1, smm2, static_cast<char>(mode) ) >= gram_len;
}

// for debug
void print_trie(TrieNode* node_ptr)
{
    if ((node_ptr->node_state) & 1) printf("end: %s|\n", node_ptr->str);
    if ((node_ptr->node_state) & 2) printf("cond: %s|%s\n", node_ptr->str, node_ptr->cond_str);
    for (int i = 0; i < node_ptr->size; ++i)
        print_trie(node_ptr->child[i].ptr);
           
}

void query_in_block(int block_idx)
{
    for (int k = block_interval[block_idx].first; k < block_interval[block_idx].last; ++ k) {
        if (read_buffer[k - 1] != ' ') continue;

        TrieNode* current;
        int j = k;
        if (read_buffer[k + 1] != ' ') {
            current = trie_entry[(unsigned char)read_buffer[k]][(unsigned char)read_buffer[k + 1]];
            j += 2;
        } else {
            int tree_idx = calc_tree_idx(read_buffer + k);
            current = trie_tree[tree_idx].root;
        }

        while (true) {
            int cond_len = 0;
            if ((current->node_state) & 2) { // pass condense node.
                cond_len = current->cond_len;
                if (!isMatchCondNode(read_buffer + j, current->cond_str, cond_len))
                    break;
            }
            if ((current->node_state) & 1) {  
                if (read_buffer[j + cond_len] == ' ')  
                    match_block[block_idx][match_block_count[block_idx]++] = Match(k,  j + cond_len, current);
            }
            j += cond_len;
            current = current->find(read_buffer[j]);
            if (current == NULL) break;
            j++;
        }        
    }
}

class ConcurrentQueryProcessor
{
public:
    void operator() (const blocked_range<size_t> &r) const
    {
        for (size_t i=r.begin();i!=r.end();++i)
        {
            query_in_block(i);
        }
    }
    ConcurrentQueryProcessor(){}
};

// struct timeval time_start;
// struct timeval time_end;
void Process()
{
    
    Ready();
    int global_query_idx = 0;
    int batches = 0;
    // double read_time = 0.0, write_time = 0.0, insert_time = 0.0, query_time = 0.0, other_time = 0.0, update_time = 0.0;
    while(true) {

        query_cnt = 0;
        batches += 1;
        update_cnt = 0;

        // gettimeofday(&time_start, NULL);
        int status = ReadBatchOperation();
        if (!status) break; 
        // gettimeofday(&time_end, NULL);
        // read_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0;

        // gettimeofday(&time_start, NULL);
        parallel_for(blocked_range<size_t>(0, TRIE_TREE_NUM),
                    ConcurrentInsertProcessor());       
        // gettimeofday(&time_end, NULL);
        // insert_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0;

        int prev_state_cnt = 0;
        for (int i = 0; i < update_cnt; ++i) {
            if (update_list[i].node->pos == 0) {
                update_list[i].node->pos = ++prev_state_cnt;
                prev_state[prev_state_cnt] = PrevNodeState(update_list[i].node, update_list[i].node->node_state);
            }
            if (update_list[i].op == 1) {
                update_list[i].node->node_state |= 1;
            }
        }

        // gettimeofday(&time_start, NULL);
        int match_cnt = 0;      

        split_into_block();

        memset(match_block_count, 0, sizeof(int) * block_interval_cnt);
        // gettimeofday(&time_end, NULL);
        // other_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0;

        // gettimeofday(&time_start, NULL);
        parallel_for(blocked_range<size_t>(0, block_interval_cnt),
                ConcurrentQueryProcessor());
        // gettimeofday(&time_end, NULL);
        // query_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0;

        // gettimeofday(&time_start, NULL);  
        for (int i = 0; i < block_interval_cnt; ++i) {
            for (int j = 0; j < match_block_count[i]; ++j) {
                matches[match_cnt++] = match_block[i][j];
            }
        }
        matches[match_cnt++] = Match(read_buffer_len, read_buffer_len, NULL);

        for (int i = 1; i <= prev_state_cnt; ++i)
            prev_state[i].node->node_state = prev_state[i].node_state;
        
        // gettimeofday(&time_end, NULL);
        // other_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0;

        // gettimeofday(&time_start, NULL);
        int cur_match_idx = 0, cur_update_idx = 0;
        for (int i = 0; i < query_cnt; ++i) {
            while (cur_update_idx < update_cnt &&
                    update_list[cur_update_idx].timestamp < query_list[i].timestamp) {
                if (update_list[cur_update_idx].op == 1)
                    update_list[cur_update_idx].node->node_state |= 1;
                else
                    update_list[cur_update_idx].node->node_state &= 2;
                cur_update_idx++;
            }

            global_query_idx++;

            int valid_match_cnt = 0;
            while (matches[cur_match_idx].begin <= query_list[i].end) {
                TrieNode* match_node_ptr = matches[cur_match_idx].node_ptr;
                if (((match_node_ptr->node_state) & 1) == 0 ||
                    match_node_ptr->global_query_idx == global_query_idx) {
                    cur_match_idx++;
                    continue;
                }
                match_node_ptr->global_query_idx = global_query_idx;
                valid_matches[valid_match_cnt++] = matches[cur_match_idx];
                cur_match_idx++;
            }
            for (int x = 0; x < valid_match_cnt; ++x) {
                if (x != 0) putchar_unlocked('|');
                // fwrite(read_buffer + valid_matches[x].begin, 1, valid_matches[x].end - valid_matches[x].begin, stdout);
                for (int j = valid_matches[x].begin; j != valid_matches[x].end; ++j)
                    putchar_unlocked(read_buffer[j]);
            }
            if (valid_match_cnt == 0) fputs_unlocked("-1", stdout);
            putchar_unlocked('\n');
        }
        cout.flush();       

        // gettimeofday(&time_end, NULL);
        // write_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0;
        
        // gettimeofday(&time_start, NULL);
        while (cur_update_idx < update_cnt) {
            if (update_list[cur_update_idx].op == 1)
                update_list[cur_update_idx].node->node_state |= 1;
            else
                update_list[cur_update_idx].node->node_state &= 2;
            cur_update_idx++;
        }
        
        for (int i = 1; i <= prev_state_cnt; ++i) {
            prev_state[i].node->pos = 0;
        }
        // gettimeofday(&time_end, NULL);
        // update_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0; 
    }
    // cerr << "read_time " << (long)read_time << "ms" << endl;
    // cerr << "write_time " << (long)write_time << "ms" <<  endl;
    // cerr << "insert_time " << (long)insert_time << "ms" <<  endl;  
    // cerr << "query_time " << (long)query_time << "ms" <<  endl;
    // cerr << "other_time " << (long)other_time << "ms" <<  endl;
    // cerr << "update_time " << (long)update_time << "ms" <<  endl;
}

void Init()
{
    tbb::task_scheduler_init init(THREADNUM);
    for (int i = 0; i < TRIE_TREE_NUM; ++i)
        trie_tree[i].init();
    input_word = new char[GRAM_TOTAL_LEN];
    read_buffer = (char *)malloc((size_t)QUERY_DOC_LEN * sizeof(char));
    stdin_buf = new char[QUERY_DOC_LEN];
    stdout_buf = new char[BUFFER_LEN];
    AllocMem();
}

int main(void)
{
    Init();
    // fcntl(0, F_SETPIPE_SZ, BUFFER_LEN);
    setvbuf(stdin, stdin_buf, _IOFBF, QUERY_DOC_LEN);
    setvbuf(stdout, stdout_buf, _IOFBF, BUFFER_LEN);
    ReadInit();
    Process();
    
    return 0;
}