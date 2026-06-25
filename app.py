import os
import re
import json
from flask import Flask, render_template, request, send_from_directory, redirect, jsonify  # pyre-ignore
from main import IRSystem  # pyre-ignore

# Inisialisasi Flask
app = Flask(__name__)

# Inisialisasi IR System (sekali saat startup)
dataset_path = os.path.join(os.path.dirname(__file__), "docs")
ir_system = IRSystem(dataset_path)
ir_system.initialize()


@app.route("/docs/<filename>")
def download_file(filename):
    return send_from_directory(dataset_path, filename)


def calculate_metrics(retrieved_list, ground_truth):
    if not ground_truth:
        return {
            'top1_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'map': 0.0
        }
    
    gt_set = set(ground_truth)
    
    # Top-1 Accuracy
    top1_accuracy = 1.0 if retrieved_list and retrieved_list[0] in gt_set else 0.0
    
    # Precision@5
    top_5 = retrieved_list[:5]
    relevant_retrieved = [doc for doc in top_5 if doc in gt_set]
    precision = len(relevant_retrieved) / 5.0
    
    # Recall@5
    recall = len(relevant_retrieved) / len(gt_set)
    
    # F1-Score
    if precision + recall > 0:
        f1_score = (2.0 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    # MAP (Average Precision over retrieved list)
    ap = 0.0
    relevant_count = 0
    for k, doc in enumerate(retrieved_list):
        if doc in gt_set:
            relevant_count += 1
            precision_at_k = relevant_count / (k + 1)
            ap += precision_at_k
    ap /= len(gt_set)
    
    return {
        'top1_accuracy': round(top1_accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'map': round(ap, 4)
    }


@app.route("/", methods=["GET", "POST"])
def index():
    data = {
        'query': '',
        'searched': False,
        'total_docs': len(ir_system.doc_names),
        'doc_names': ir_system.doc_names,
        'results': [],
        'relevant_count': 0,
        'query_preprocessing': None,
        'doc_preprocessing': [],
        'feature_data': None,
        'comparison': None,
        'full_vectors': None,
        'method_comparison': None,
        'word2vec_sg_training': {},
        'word2vec_cbow_training': {},
        'word2vec_ft_training': {},
        'sg_results': [],
        'cbow_results': [],
        'ft_results': [],
        'sim_matrix': [],
        'w2v_sg_sim_matrix': [],
        'w2v_cbow_sim_matrix': [],
        'w2v_ft_sim_matrix': [],
        'sim_doc_names': [],
        'current_version': ir_system.current_version,
        'versions': ir_system.list_versions(),
        'ground_truth_docs': [],
        'has_ground_truth': False,
        'eval_metrics': {},
        'global_eval_metrics': {},
        'query_history': []
    }

    # Load riwayat pencarian awal
    data['query_history'] = ir_system.load_query_history()

    query = ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
    else:
        query = request.args.get("query", "").strip()

    if query:
        data['query'] = query
        data['searched'] = True

        # Load riwayat pencarian
        data['query_history'] = ir_system.load_query_history()

        # Jalankan pencarian
        search_result = ir_system.search(query)

        # Update data untuk template
        data.update(search_result)
        data['word2vec_sg_training'] = search_result['method_comparison'].get(
            'word2vec_sg_training',
            {}
        )
        data['word2vec_cbow_training'] = search_result['method_comparison'].get(
            'word2vec_cbow_training',
            {}
        )
        data['word2vec_ft_training'] = search_result['method_comparison'].get(
            'word2vec_ft_training',
            {}
        )
        data['current_version'] = ir_system.current_version
        data['versions'] = ir_system.list_versions()

        # Load ground truth dari storage
        gt_file = os.path.join("storage", "ground_truth.json")
        ground_truth_docs = []
        has_ground_truth = False
        if os.path.exists(gt_file):
            try:
                with open(gt_file, "r") as f:
                    gt_data = json.load(f)
                    if query in gt_data:
                        ground_truth_docs = gt_data[query]
                        has_ground_truth = True
            except:
                pass
        
        data['ground_truth_docs'] = ground_truth_docs
        data['has_ground_truth'] = has_ground_truth

        # Hitung performance evaluation metrics
        tfidf_retrieved = [r['filename'] for r in search_result['results']]
        sg_retrieved = [r['filename'] for r in search_result['sg_results']]
        cbow_retrieved = [r['filename'] for r in search_result['cbow_results']]
        ft_retrieved = [r['filename'] for r in search_result['ft_results']]

        gt_set = set(ground_truth_docs)
        data['eval_metrics'] = {
            'tfidf': calculate_metrics(tfidf_retrieved, gt_set),
            'sg': calculate_metrics(sg_retrieved, gt_set),
            'cbow': calculate_metrics(cbow_retrieved, gt_set),
            'ft': calculate_metrics(ft_retrieved, gt_set)
        }

    return render_template("index.html", **data)


@app.route("/evaluation")
def evaluation():
    global_eval_metrics = ir_system.evaluate_global_benchmark()
    global_eval_details = ir_system.get_global_benchmark_details()
    query_history = ir_system.load_query_history()
    
    # Calculate top-1 success count for each model for the Donut Chart
    gt_file = os.path.join("storage", "ground_truth.json")
    success_counts = {'tfidf': 0, 'cbow': 0, 'sg': 0, 'ft': 0}
    
    if os.path.exists(gt_file):
        try:
            with open(gt_file, "r") as f:
                gt_data = json.load(f)
        except:
            gt_data = {}
            
        for query, gt_docs in gt_data.items():
            if not gt_docs:
                continue
            # Run search silently to get retrieved rankings
            res = ir_system.search(query, silent=True)
            
            # Helper to check if top-1 is in ground truth
            def is_correct(results):
                return results and results[0]['filename'] in gt_docs
                
            if is_correct(res.get('results')):
                success_counts['tfidf'] += 1
            if is_correct(res.get('cbow_results')):
                success_counts['cbow'] += 1
            if is_correct(res.get('sg_results')):
                success_counts['sg'] += 1
            if is_correct(res.get('ft_results')):
                success_counts['ft'] += 1
                
    data = {
        'total_docs': len(ir_system.doc_names),
        'current_version': ir_system.current_version,
        'versions': ir_system.list_versions(),
        'global_eval_metrics': global_eval_metrics,
        'global_eval_details': global_eval_details,
        'query_history': query_history,
        'success_counts': success_counts,
        'total_queries': len(global_eval_details)
    }
    
    return render_template("evaluation.html", **data)


@app.route("/save_ground_truth", methods=["POST"])
def save_ground_truth():
    query = request.form.get("query", "").strip()
    relevant_docs = request.form.getlist("relevant_docs")
    
    gt_file = os.path.join("storage", "ground_truth.json")
    gt_data = {}
    if os.path.exists(gt_file):
        try:
            with open(gt_file, "r") as f:
                gt_data = json.load(f)
        except:
            pass
            
    gt_data[query] = relevant_docs
    
    os.makedirs("storage", exist_ok=True)
    with open(gt_file, "w") as f:
        json.dump(gt_data, f)
        
    if query:
        ir_system.add_to_query_history(query)
        
    return redirect(f"/?query={query}")


@app.route("/select_version", methods=["POST"])
def select_version():
    version_name = request.form.get("version_name")
    if version_name:
        ir_system.load_version(version_name)
    return redirect("/")


@app.route("/train_version", methods=["POST"])
def train_version():
    version_name = request.form.get("version_name", "").strip()
    # sanitasi nama versi agar aman untuk direktori
    version_name = re.sub(r'[^a-zA-Z0-9_\-]', '', version_name)
    if version_name:
        architecture = request.form.get("architecture", "skipgram").strip()
        implementation_mode = request.form.get("implementation_mode", "library").strip()
        use_library = (implementation_mode == "library")
        window_size = int(request.form.get("window_size", 2))
        vector_size = int(request.form.get("vector_size", 10))
        epochs = int(request.form.get("epochs", 1))
        max_training_pairs = int(request.form.get("max_training_pairs", 300))
        
        ir_system.initialize(
            force_fresh=True,
            architecture=architecture,
            window_size=window_size,
            vector_size=vector_size,
            epochs=epochs,
            max_training_pairs=max_training_pairs,
            use_library=use_library
        )
        ir_system.save_version(version_name)
    return redirect("/")


@app.route("/download_one_hot/<version_name>", methods=["GET"])
def download_one_hot(version_name):
    version_name = re.sub(r'[^a-zA-Z0-9_\-]', '', version_name)
    versions_dir = os.path.join(os.path.dirname(__file__), "storage", "versions", version_name)
    one_hot_file = os.path.join(versions_dir, "one_hot_encodings.json")
    if os.path.exists(one_hot_file):
        return send_from_directory(versions_dir, "one_hot_encodings.json", as_attachment=True)
    return "File one-hot encoding untuk versi ini tidak ditemukan.", 404


@app.route("/api/benchmark_details")
def api_benchmark_details():
    global_eval_details = ir_system.get_global_benchmark_details()
    return jsonify(global_eval_details)


@app.route("/delete_history", methods=["POST"])
def delete_history():
    query = request.json.get("query", "").strip()
    if query:
        ir_system.remove_from_query_history(query)
    return jsonify({"status": "success"})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    ir_system.clear_query_history()
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
