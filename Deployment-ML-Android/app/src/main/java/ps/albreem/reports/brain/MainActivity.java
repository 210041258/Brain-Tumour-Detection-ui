package ps.albreem.reports.brain;

import android.animation.*;
import android.app.*;
import android.content.*;
import android.content.res.*;
import android.graphics.*;
import android.graphics.drawable.*;
import android.media.*;
import android.net.*;
import android.os.*;
import android.text.*;
import android.text.style.*;
import android.util.*;
import android.view.*;
import android.view.View;
import android.view.View.*;
import android.view.animation.*;
import android.webkit.*;
import android.widget.*;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.BaseAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.annotation.*;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import com.bumptech.glide.Glide;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.firebase.FirebaseApp;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.GenericTypeIndicator;
import com.google.firebase.database.ValueEventListener;
import java.io.*;
import java.text.*;
import java.util.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.regex.*;
import org.json.*;

public class MainActivity extends AppCompatActivity {
	
	private FirebaseDatabase _firebase = FirebaseDatabase.getInstance();
	
	private FloatingActionButton _fab;
	private boolean dashboard_bool = false;
	private boolean home_bool = false;
	private boolean search_bool = false;
	
	private ArrayList<HashMap<String, Object>> map = new ArrayList<>();
	
	private ScrollView vscroll1;
	private LinearLayout linear2;
	private LinearLayout linear3;
	private ListView dashboard;
	private LinearLayout linear16;
	private LinearLayout search_layout;
	private ImageView imageview1;
	private LinearLayout linear4;
	private ImageView imageview2;
	private LinearLayout linear5;
	private ImageView imageview3;
	private LinearLayout linear6;
	private ListView home;
	private LinearLayout linear7;
	private LinearLayout linear8;
	private ImageView imageview4;
	private EditText edittext1;
	private LinearLayout linear14;
	private ImageView imageview5;
	private LinearLayout linear15;
	private LinearLayout linear9;
	private LinearLayout linear10;
	private LinearLayout linear11;
	private LinearLayout linear12;
	private LinearLayout linear13;
	private TextView textview1;
	private TextView textview6;
	private TextView textview2;
	private TextView textview7;
	private TextView textview3;
	private TextView textview8;
	private TextView textview4;
	private TextView textview9;
	private TextView textview5;
	
	private DatabaseReference predication = _firebase.getReference("predication");
	private ChildEventListener _predication_child_listener;
	
	@Override
	protected void onCreate(Bundle _savedInstanceState) {
		super.onCreate(_savedInstanceState);
		setContentView(R.layout.main);
		initialize(_savedInstanceState);
		FirebaseApp.initializeApp(this);
		initializeLogic();
	}
	
	private void initialize(Bundle _savedInstanceState) {
		_fab = findViewById(R.id._fab);
		
		vscroll1 = findViewById(R.id.vscroll1);
		linear2 = findViewById(R.id.linear2);
		linear3 = findViewById(R.id.linear3);
		dashboard = findViewById(R.id.dashboard);
		linear16 = findViewById(R.id.linear16);
		search_layout = findViewById(R.id.search_layout);
		imageview1 = findViewById(R.id.imageview1);
		linear4 = findViewById(R.id.linear4);
		imageview2 = findViewById(R.id.imageview2);
		linear5 = findViewById(R.id.linear5);
		imageview3 = findViewById(R.id.imageview3);
		linear6 = findViewById(R.id.linear6);
		home = findViewById(R.id.home);
		linear7 = findViewById(R.id.linear7);
		linear8 = findViewById(R.id.linear8);
		imageview4 = findViewById(R.id.imageview4);
		edittext1 = findViewById(R.id.edittext1);
		linear14 = findViewById(R.id.linear14);
		imageview5 = findViewById(R.id.imageview5);
		linear15 = findViewById(R.id.linear15);
		linear9 = findViewById(R.id.linear9);
		linear10 = findViewById(R.id.linear10);
		linear11 = findViewById(R.id.linear11);
		linear12 = findViewById(R.id.linear12);
		linear13 = findViewById(R.id.linear13);
		textview1 = findViewById(R.id.textview1);
		textview6 = findViewById(R.id.textview6);
		textview2 = findViewById(R.id.textview2);
		textview7 = findViewById(R.id.textview7);
		textview3 = findViewById(R.id.textview3);
		textview8 = findViewById(R.id.textview8);
		textview4 = findViewById(R.id.textview4);
		textview9 = findViewById(R.id.textview9);
		textview5 = findViewById(R.id.textview5);
		
		imageview1.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View _view) {
				dashboard_bool = false;
				search_bool = false;
				home_bool = true;
				home.setVisibility(View.VISIBLE);
				dashboard.setVisibility(View.GONE);
				search_layout.setVisibility(View.GONE);
				_fab.show();
				predication.addListenerForSingleValueEvent(new ValueEventListener() {
					@Override
					public void onDataChange(DataSnapshot _dataSnapshot) {
						map = new ArrayList<>();
						try {
							GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
							for (DataSnapshot _data : _dataSnapshot.getChildren()) {
								HashMap<String, Object> _map = _data.getValue(_ind);
								map.add(_map);
							}
						}
						catch (Exception _e) {
							_e.printStackTrace();
						}
						home.setAdapter(new HomeAdapter(map));
						((BaseAdapter)home.getAdapter()).notifyDataSetChanged();
					}
					@Override
					public void onCancelled(DatabaseError _databaseError) {
					}
				});
			}
		});
		
		imageview2.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View _view) {
				dashboard_bool = true;
				search_bool = false;
				home_bool = false;
				home.setVisibility(View.GONE);
				dashboard.setVisibility(View.VISIBLE);
				search_layout.setVisibility(View.GONE);
				_fab.show();
				predication.addListenerForSingleValueEvent(new ValueEventListener() {
					@Override
					public void onDataChange(DataSnapshot _dataSnapshot) {
						map = new ArrayList<>();
						try {
							GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
							for (DataSnapshot _data : _dataSnapshot.getChildren()) {
								HashMap<String, Object> _map = _data.getValue(_ind);
								map.add(_map);
							}
						}
						catch (Exception _e) {
							_e.printStackTrace();
						}
						dashboard.setAdapter(new DashboardAdapter(map));
						((BaseAdapter)dashboard.getAdapter()).notifyDataSetChanged();
					}
					@Override
					public void onCancelled(DatabaseError _databaseError) {
					}
				});
			}
		});
		
		imageview3.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View _view) {
				dashboard_bool = false;
				search_bool = true;
				home_bool = false;
				home.setVisibility(View.GONE);
				dashboard.setVisibility(View.GONE);
				search_layout.setVisibility(View.VISIBLE);
				_fab.hide();
			}
		});
		
		home.setOnItemClickListener(new AdapterView.OnItemClickListener() {
			@Override
			public void onItemClick(AdapterView<?> _param1, View _param2, int _param3, long _param4) {
				final int _position = _param3;
				
			}
		});
		
		imageview4.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View _view) {
				String id = edittext1.getText().toString().trim();
				
				DatabaseReference ref = FirebaseDatabase.getInstance().getReference("predication").child(id);
				
				ref.addListenerForSingleValueEvent(new ValueEventListener() {
					    @Override
					    public void onDataChange(@NonNull DataSnapshot snapshot) {
						        if (snapshot.exists()) {
							            String imageUrl = snapshot.child("url").getValue(String.class);
							            String returnedId = snapshot.child("id").getValue(String.class);
							            String filename = snapshot.child("filename").getValue(String.class);
							            String clinical = snapshot.child("extra").getValue(String.class);
							            String suggestion = snapshot.child("analysis3").getValue(String.class); // using same as first form
							            String details = snapshot.child("analysis2").getValue(String.class);    // using same as first form
							
							            // Load image using Picasso
							            
							Glide.with(getApplicationContext()).load(Uri.parse(imageUrl)).into(imageview5);
							
							            // Set texts (structured like the first form, but clearer)
							            textview1.setText("Filename: " + filename);
							            textview3.setText("Suggestion: " + suggestion);
							            textview4.setText("Details: " + details);
							            textview5.setText("ID: " + returnedId); // Additional display if needed
							            textview6.setText("Summary: " + clinical); 
							
							        } else {
							            Toast.makeText(MainActivity.this, "ID not found in prediction branch", Toast.LENGTH_SHORT).show();
							        }
						    }
					
					    @Override
					    public void onCancelled(@NonNull DatabaseError error) {
						        Toast.makeText(MainActivity.this, "Error: " + error.getMessage(), Toast.LENGTH_SHORT).show();
						    }
				});
			}
		});
		
		imageview5.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View _view) {
				if (false) {
					Glide.with(getApplicationContext()).load(Uri.parse("b")).into(imageview5);
				}
			}
		});
		
		_fab.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View _view) {
				if (home_bool) {
					predication.addListenerForSingleValueEvent(new ValueEventListener() {
						@Override
						public void onDataChange(DataSnapshot _dataSnapshot) {
							map = new ArrayList<>();
							try {
								GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
								for (DataSnapshot _data : _dataSnapshot.getChildren()) {
									HashMap<String, Object> _map = _data.getValue(_ind);
									map.add(_map);
								}
							}
							catch (Exception _e) {
								_e.printStackTrace();
							}
							home.setAdapter(new HomeAdapter(map));
							((BaseAdapter)home.getAdapter()).notifyDataSetChanged();
						}
						@Override
						public void onCancelled(DatabaseError _databaseError) {
						}
					});
				}
				else {
					predication.addListenerForSingleValueEvent(new ValueEventListener() {
						@Override
						public void onDataChange(DataSnapshot _dataSnapshot) {
							map = new ArrayList<>();
							try {
								GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
								for (DataSnapshot _data : _dataSnapshot.getChildren()) {
									HashMap<String, Object> _map = _data.getValue(_ind);
									map.add(_map);
								}
							}
							catch (Exception _e) {
								_e.printStackTrace();
							}
							dashboard.setAdapter(new DashboardAdapter(map));
							((BaseAdapter)dashboard.getAdapter()).notifyDataSetChanged();
						}
						@Override
						public void onCancelled(DatabaseError _databaseError) {
						}
					});
				}
			}
		});
		
		_predication_child_listener = new ChildEventListener() {
			@Override
			public void onChildAdded(DataSnapshot _param1, String _param2) {
				GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
				final String _childKey = _param1.getKey();
				final HashMap<String, Object> _childValue = _param1.getValue(_ind);
				
			}
			
			@Override
			public void onChildChanged(DataSnapshot _param1, String _param2) {
				GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
				final String _childKey = _param1.getKey();
				final HashMap<String, Object> _childValue = _param1.getValue(_ind);
				
			}
			
			@Override
			public void onChildMoved(DataSnapshot _param1, String _param2) {
				
			}
			
			@Override
			public void onChildRemoved(DataSnapshot _param1) {
				GenericTypeIndicator<HashMap<String, Object>> _ind = new GenericTypeIndicator<HashMap<String, Object>>() {};
				final String _childKey = _param1.getKey();
				final HashMap<String, Object> _childValue = _param1.getValue(_ind);
				
			}
			
			@Override
			public void onCancelled(DatabaseError _param1) {
				final int _errorCode = _param1.getCode();
				final String _errorMessage = _param1.getMessage();
				
			}
		};
		predication.addChildEventListener(_predication_child_listener);
	}
	
	private void initializeLogic() {
		dashboard_bool = false;
		search_bool = false;
		home_bool = true;
		home.setVisibility(View.VISIBLE);
		dashboard.setVisibility(View.GONE);
		search_layout.setVisibility(View.GONE);
	}
	
	public class DashboardAdapter extends BaseAdapter {
		
		ArrayList<HashMap<String, Object>> _data;
		
		public DashboardAdapter(ArrayList<HashMap<String, Object>> _arr) {
			_data = _arr;
		}
		
		@Override
		public int getCount() {
			return _data.size();
		}
		
		@Override
		public HashMap<String, Object> getItem(int _index) {
			return _data.get(_index);
		}
		
		@Override
		public long getItemId(int _index) {
			return _index;
		}
		
		@Override
		public View getView(final int _position, View _v, ViewGroup _container) {
			LayoutInflater _inflater = getLayoutInflater();
			View _view = _v;
			if (_view == null) {
				_view = _inflater.inflate(R.layout.dashboard, null);
			}
			
			final LinearLayout linear2 = _view.findViewById(R.id.linear2);
			final LinearLayout linear3 = _view.findViewById(R.id.linear3);
			final LinearLayout linear4 = _view.findViewById(R.id.linear4);
			final LinearLayout linear5 = _view.findViewById(R.id.linear5);
			final LinearLayout linear6 = _view.findViewById(R.id.linear6);
			final LinearLayout linear7 = _view.findViewById(R.id.linear7);
			final TextView textview1 = _view.findViewById(R.id.textview1);
			final TextView textview2 = _view.findViewById(R.id.textview2);
			final TextView textview3 = _view.findViewById(R.id.textview3);
			final TextView textview4 = _view.findViewById(R.id.textview4);
			final TextView textview11 = _view.findViewById(R.id.textview11);
			final TextView textview5 = _view.findViewById(R.id.textview5);
			final TextView textview6 = _view.findViewById(R.id.textview6);
			final TextView textview7 = _view.findViewById(R.id.textview7);
			final TextView textview10 = _view.findViewById(R.id.textview10);
			final TextView textview8 = _view.findViewById(R.id.textview8);
			final TextView textview9 = _view.findViewById(R.id.textview9);
			
			textview2.setText(_data.get((int)_position).get("filename").toString());
			textview4.setText(_data.get((int)_position).get("id").toString());
			textview6.setText(_data.get((int)_position).get("extra").toString());
			textview10.setText(_data.get((int)_position).get("analysis2").toString());
			textview9.setText(_data.get((int)_position).get("analysis3").toString());
			
			return _view;
		}
	}
	
	public class HomeAdapter extends BaseAdapter {
		
		ArrayList<HashMap<String, Object>> _data;
		
		public HomeAdapter(ArrayList<HashMap<String, Object>> _arr) {
			_data = _arr;
		}
		
		@Override
		public int getCount() {
			return _data.size();
		}
		
		@Override
		public HashMap<String, Object> getItem(int _index) {
			return _data.get(_index);
		}
		
		@Override
		public long getItemId(int _index) {
			return _index;
		}
		
		@Override
		public View getView(final int _position, View _v, ViewGroup _container) {
			LayoutInflater _inflater = getLayoutInflater();
			View _view = _v;
			if (_view == null) {
				_view = _inflater.inflate(R.layout.home, null);
			}
			
			final LinearLayout linear2 = _view.findViewById(R.id.linear2);
			final LinearLayout linear1 = _view.findViewById(R.id.linear1);
			final LinearLayout linear3 = _view.findViewById(R.id.linear3);
			final LinearLayout linear4 = _view.findViewById(R.id.linear4);
			final TextView textview4 = _view.findViewById(R.id.textview4);
			final TextView textview1 = _view.findViewById(R.id.textview1);
			final TextView textview5 = _view.findViewById(R.id.textview5);
			final TextView textview2 = _view.findViewById(R.id.textview2);
			final TextView textview6 = _view.findViewById(R.id.textview6);
			final TextView textview3 = _view.findViewById(R.id.textview3);
			
			textview1.setText(_data.get((int)_position).get("filename").toString());
			textview2.setText(_data.get((int)_position).get("id").toString());
			textview3.setText(_data.get((int)_position).get("details").toString());
			
			return _view;
		}
	}
	
	@Deprecated
	public void showMessage(String _s) {
		Toast.makeText(getApplicationContext(), _s, Toast.LENGTH_SHORT).show();
	}
	
	@Deprecated
	public int getLocationX(View _v) {
		int _location[] = new int[2];
		_v.getLocationInWindow(_location);
		return _location[0];
	}
	
	@Deprecated
	public int getLocationY(View _v) {
		int _location[] = new int[2];
		_v.getLocationInWindow(_location);
		return _location[1];
	}
	
	@Deprecated
	public int getRandom(int _min, int _max) {
		Random random = new Random();
		return random.nextInt(_max - _min + 1) + _min;
	}
	
	@Deprecated
	public ArrayList<Double> getCheckedItemPositionsToArray(ListView _list) {
		ArrayList<Double> _result = new ArrayList<Double>();
		SparseBooleanArray _arr = _list.getCheckedItemPositions();
		for (int _iIdx = 0; _iIdx < _arr.size(); _iIdx++) {
			if (_arr.valueAt(_iIdx))
			_result.add((double)_arr.keyAt(_iIdx));
		}
		return _result;
	}
	
	@Deprecated
	public float getDip(int _input) {
		return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, _input, getResources().getDisplayMetrics());
	}
	
	@Deprecated
	public int getDisplayWidthPixels() {
		return getResources().getDisplayMetrics().widthPixels;
	}
	
	@Deprecated
	public int getDisplayHeightPixels() {
		return getResources().getDisplayMetrics().heightPixels;
	}
}