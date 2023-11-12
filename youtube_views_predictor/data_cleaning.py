import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    
    def __init__(self, trending_data, category_id):
        
        self.trending_data_df = trending_data
        self.cat_df = category_id
        
        self.std_scaler_views = StandardScaler()
        self.std_scaler_likes = StandardScaler()
        self.std_scaler_dislikes = StandardScaler()


    def clean_transform(self):
        
        # Drop Duplicates
        self.trending_data_df.drop_duplicates(inplace=True)
        
        # Expand Category ID JSON Dataframe
        self.cat_df = pd.DataFrame.from_dict(
            [cat_dict for cat_dict in self.cat_df["items"]]
        ).drop(["kind"], axis=1)
        
        self.cat_df["title"] = self.cat_df["snippet"].apply(lambda x: x["title"])
        self.cat_df["assignable"] = self.cat_df["snippet"].apply(lambda x: x["assignable"])
        #self.cat_df["channelId"] = self.cat_df["snippet"].apply(lambda x: x["channelId"])
        self.cat_df.drop(["snippet"], axis=1, inplace=True)
        
        # Split tags
        self.trending_data_df["tags"] = self.trending_data_df["tags"].apply(
            lambda x: x.split("|") if x != "[None]" else None
        )
                
        # Cast category id
        self.cat_df["id"] = self.cat_df.id.astype("int")

        # Category ids in df are in the category id JSON
        #cat_diff = set(self.trending_data_df.categoryId.unique()) - set(self.cat_df.id)
        #if len(cat_diff) > 0:
        #    print(f"Found a category id that is not in the JSON: {cat_diff}")
        
        # Removing non-ASCII
        self.trending_data_df.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        self.trending_data_df.description.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        
        
        
    def tranform_views(self, method: str, fit: bool)
        
        """
        method can be log, standard scaler, or both
        """
        self.trending_data_df["view_count_scaled"] = self.trending_data_df["view_count"]
        self.trending_data_df["likes_scaled"] = self.trending_data_df["likes"]
        self.trending_data_df["dislikes_scaled"] = self.trending_data_df["dislikes"]
        
        
        if method in ["log", "both"]:
            self.trending_data_df["view_count_scaled"] = np.log(self.trending_data_df["view_count"])
            self.trending_data_df["likes_scaled"] = np.log(self.trending_data_df["likes"])
            self.trending_data_df["dislikes_scaled"] = np.log(self.trending_data_df["dislikes"])
        
        if method in ["StandardScaler", "both"] and fit:
            self.trending_data_df["view_count_scaled"] = self.std_scaler_views.fit_transform(
                self.trending_data_df.view_count.values.reshape(-1,1)
            )
            self.trending_data_df["likes_scaled"] = self.std_scaler_views.fit_transform(
                self.trending_data_df.likes.values.reshape(-1,1)
            )
            self.trending_data_df["dislikes_scaled"] = self.std_scaler_views.fit_transform(
                self.trending_data_df.dislikes.values.reshape(-1,1)
            )
            
        if method in ["StandardScaler", "both"] and not fit:
            self.trending_data_df["view_count_scaled"] = self.std_scaler_views.transform(
                self.trending_data_df.view_count.values.reshape(-1,1)
            )
            self.trending_data_df["likes_scaled"] = self.std_scaler_views.transform(
                self.trending_data_df.likes.values.reshape(-1,1)
            )
            self.trending_data_df["dislikes_scaled"] = self.std_scaler_views.transform(
                self.trending_data_df.dislikes.values.reshape(-1,1)
            )
        
        
        
    def save_cleaned(self, trending_data_path, category_id_path):
        self.trending_data_df.to_csv(trending_data_path, index=False)
        self.cat_df.to_csv(category_id_path, index=False)
        
